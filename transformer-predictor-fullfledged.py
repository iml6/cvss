"""
CVE CVSS Score Prediction System using Transformer Models
Fine-tunes BERT for multi-label classification of CVSS metrics
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyTorch and Transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    logging as hf_logging
)

# Data processing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
import requests
from tqdm import tqdm
import pickle

# Set logging level
hf_logging.set_verbosity_error()

@dataclass
class CVSSConfig:
    """Configuration for CVSS prediction model"""
    # Model selection - security-specific models perform better
    model_name: str = "microsoft/codebert-base"  # Options: "jackaduma/SecBERT", "microsoft/deberta-v3-base"
    max_length: int = 512
    
    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1  # Warmup for 10% of training steps
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    gradient_accumulation_steps: int = 1  # Increase for larger effective batch size
    
    # Advanced training techniques
    use_class_weights: bool = True  # Balance imbalanced classes
    use_focal_loss: bool = True  # Better handling of hard examples
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_mixup: bool = True  # Data augmentation
    mixup_alpha: float = 0.2
    label_smoothing: float = 0.1  # Prevent overconfidence
    
    # Model architecture enhancements
    use_lstm: bool = True  # Add LSTM layer for sequence modeling
    lstm_hidden_size: int = 256
    num_lstm_layers: int = 2
    use_crf: bool = False  # Conditional Random Fields for metric dependencies
    freeze_embeddings: bool = False  # Freeze BERT embeddings initially
    unfreeze_after_epoch: int = 3  # Unfreeze embeddings after N epochs
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    
    # Training optimization
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    min_lr: float = 1e-6
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CVSS v3.1 metrics
    cvss_metrics: Dict = None
    
    def __post_init__(self):
        self.cvss_metrics = {
            'attackVector': ['NETWORK', 'ADJACENT_NETWORK', 'LOCAL', 'PHYSICAL'],
            'attackComplexity': ['LOW', 'HIGH'],
            'privilegesRequired': ['NONE', 'LOW', 'HIGH'],
            'userInteraction': ['NONE', 'REQUIRED'],
            'scope': ['UNCHANGED', 'CHANGED'],
            'confidentialityImpact': ['NONE', 'LOW', 'HIGH'],
            'integrityImpact': ['NONE', 'LOW', 'HIGH'],
            'availabilityImpact': ['NONE', 'LOW', 'HIGH']
        }


class CVEDataset(Dataset):
    """PyTorch Dataset for CVE data"""
    
    def __init__(self, texts: List[str], labels: Dict[str, np.ndarray], 
                 tokenizer, max_length: int = 512):
        """
        Initialize CVE dataset
        
        Args:
            texts: List of CVE descriptions
            labels: Dictionary of labels for each CVSS metric
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metric_names = list(labels.keys())
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels for all metrics
        label_dict = {}
        for metric in self.metric_names:
            label_dict[metric] = torch.tensor(self.labels[metric][idx], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label_dict
        }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CVSSTransformer(nn.Module):
    """
    Enhanced Transformer model for CVSS prediction with advanced architectures
    """
    
    def __init__(self, config: CVSSConfig):
        """
        Initialize the enhanced CVSS transformer model
        
        Args:
            config: Model configuration
        """
        super(CVSSTransformer, self).__init__()
        
        self.config = config
        
        # Load pre-trained transformer
        if 'deberta' in config.model_name.lower():
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(config.model_name)
        else:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Optionally freeze embeddings initially
        if config.freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # Add LSTM layers for sequence modeling
        self.use_lstm = config.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(
                hidden_size,
                config.lstm_hidden_size,
                num_layers=config.num_lstm_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.num_lstm_layers > 1 else 0,
                bidirectional=True
            )
            lstm_output_size = config.lstm_hidden_size * 2  # bidirectional
        else:
            lstm_output_size = hidden_size
        
        # Enhanced attention mechanism with multiple heads
        self.attention = nn.MultiheadAttention(
            lstm_output_size if self.use_lstm else hidden_size,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Cross-attention between different parts of the input
        self.cross_attention = nn.MultiheadAttention(
            lstm_output_size if self.use_lstm else hidden_size,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Dropout layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.feature_dropout = nn.Dropout(config.dropout_rate * 0.5)
        
        # Enhanced feature extraction with residual connections
        feature_input_size = lstm_output_size if self.use_lstm else hidden_size
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_input_size * 2, hidden_size),  # *2 for concatenated features
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.5)
        )
        
        # Metric relationship modeling - some metrics are correlated
        self.metric_interaction = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2)
        )
        
        # Separate classification heads with different architectures
        self.classifiers = nn.ModuleDict()
        
        # Different architectures for different metric types
        for metric, classes in config.cvss_metrics.items():
            if metric in ['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteraction']:
                # Exploitability metrics - use deeper network
                self.classifiers[metric] = nn.Sequential(
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rate * 0.3),
                    nn.Linear(hidden_size // 4, len(classes))
                )
            else:
                # Impact metrics - use simpler network
                self.classifiers[metric] = nn.Linear(hidden_size // 2, len(classes))
        
        # Auxiliary task: predict severity directly (helps learning)
        self.severity_classifier = nn.Linear(hidden_size // 2, 5)  # NONE, LOW, MEDIUM, HIGH, CRITICAL
        
    def unfreeze_embeddings(self):
        """Unfreeze BERT embeddings for fine-tuning"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary of logits for each CVSS metric
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use multiple layers for better representation
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        if hidden_states:
            # Weighted average of last 4 layers
            last_layers = torch.stack(hidden_states[-4:])
            weights = F.softmax(torch.tensor([0.1, 0.2, 0.3, 0.4]).to(input_ids.device), dim=0)
            sequence_output = torch.sum(last_layers * weights.view(-1, 1, 1, 1), dim=0)
        else:
            sequence_output = outputs.last_hidden_state
        
        # Apply LSTM if configured
        if self.use_lstm:
            lstm_out, _ = self.lstm(sequence_output)
            sequence_output = lstm_out
        
        # Get CLS token and apply self-attention
        cls_embedding = sequence_output[:, 0, :]
        
        # Self-attention on sequence
        attn_output, attn_weights = self.attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool() if not self.use_lstm else None
        )
        
        # Cross-attention between CLS and sequence
        cross_attn_output, _ = self.cross_attention(
            cls_embedding.unsqueeze(1),
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool() if not self.use_lstm else None
        )
        
        # Combine different representations
        pooled_output = torch.cat([
            cls_embedding,
            torch.mean(attn_output, dim=1),
            cross_attn_output.squeeze(1)
        ], dim=-1)[:, :sequence_output.size(-1) * 2]  # Ensure correct size
        
        pooled_output = self.dropout(pooled_output)
        
        # Extract shared features
        features = self.feature_layer(pooled_output)
        
        # Add metric interaction features
        interaction_features = self.metric_interaction(features)
        features = features + interaction_features  # Residual connection
        
        # Get predictions for each metric
        logits = {}
        for metric, classifier in self.classifiers.items():
            logits[metric] = classifier(features)
        
        # Auxiliary severity prediction
        logits['severity'] = self.severity_classifier(features)
        
        if return_features:
            return logits, features, attn_weights
        
        return logits


class CVSSPredictor:
    """
    Complete system for CVE CVSS prediction using transformers
    """
    
    def __init__(self, config: CVSSConfig = None):
        """
        Initialize the CVSS predictor
        
        Args:
            config: Model configuration
        """
        self.config = config or CVSSConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = CVSSTransformer(self.config).to(self.device)
        
        # Label encoders for each metric
        self.label_encoders = {}
        for metric, classes in self.config.cvss_metrics.items():
            le = LabelEncoder()
            le.fit(classes)
            self.label_encoders[metric] = le
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': {}
        }
    
    def augment_text(self, text: str) -> str:
        """
        Augment CVE description text for better generalization
        
        Args:
            text: Original text
            
        Returns:
            Augmented text
        """
        import random
        
        # Synonym replacement for security terms
        security_synonyms = {
            'vulnerability': ['weakness', 'flaw', 'security issue', 'bug'],
            'attacker': ['adversary', 'malicious user', 'threat actor'],
            'exploit': ['leverage', 'abuse', 'take advantage of'],
            'remote': ['network-based', 'distant', 'off-site'],
            'local': ['on-system', 'on-device', 'host-based'],
            'execute': ['run', 'perform', 'carry out'],
            'arbitrary': ['unauthorized', 'malicious', 'unintended'],
            'privilege': ['permission', 'access right', 'authorization'],
            'elevation': ['escalation', 'increase', 'raising']
        }
        
        # Random deletion (remove less important words)
        if random.random() < 0.3:
            words = text.split()
            if len(words) > 10:
                num_delete = random.randint(1, max(1, len(words) // 10))
                for _ in range(num_delete):
                    if len(words) > 5:
                        idx = random.randint(0, len(words) - 1)
                        if words[idx].lower() not in ['cve', 'vulnerability', 'allows', 'attacker']:
                            del words[idx]
                text = ' '.join(words)
        
        # Paraphrase by reordering clauses
        if random.random() < 0.3 and ' allows ' in text:
            parts = text.split(' allows ')
            if len(parts) == 2:
                text = f"An attacker can {parts[1]} due to {parts[0]}"
        
        return text
    
    def compute_class_weights(self, labels: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Compute class weights for handling imbalanced data
        
        Args:
            labels: Dictionary of labels for each metric
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        weights = {}
        for metric, metric_labels in labels.items():
            classes = np.unique(metric_labels)
            class_weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=metric_labels
            )
            weights[metric] = torch.tensor(class_weights, dtype=torch.float32)
        
        return weights
    
    def mixup_data(self, x, y, alpha=0.2):
        """
        Mixup augmentation for better generalization
        
        Args:
            x: Input features
            y: Labels
            alpha: Mixup parameter
            
        Returns:
            Mixed inputs, mixed targets, and lambda
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, {k: v[index] for k, v in y.items()} if isinstance(y, dict) else y[index]
        
        return mixed_x, y_a, y_b, lam
        
    def load_local_cves(self, data_path: str, years: List[int] = None, 
                        limit_per_year: int = None) -> List[Dict]:
        """
        Load CVE data from local cloned repository
        
        Args:
            data_path: Path to the cloned cvelistV5 repository (e.g., './cves')
            years: List of years to load (None for all)
            limit_per_year: Maximum CVEs per year (None for all)
            
        Returns:
            List of CVE dictionaries
        """
        cves = []
        base_path = Path(data_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"CVE data path not found: {data_path}")
        
        # Get all year directories
        if years is None:
            year_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()])
            years = [int(d.name) for d in year_dirs]
        
        print(f"Loading CVEs from years: {years}")
        
        for year in years:
            year_path = base_path / str(year)
            if not year_path.exists():
                print(f"Year {year} directory not found, skipping...")
                continue
            
            year_cves = []
            cve_files = []
            
            # Collect all JSON files in subdirectories
            for subdir in year_path.iterdir():
                if subdir.is_dir():
                    cve_files.extend(list(subdir.glob('*.json')))
            
            # Sort for consistent ordering
            cve_files.sort()
            
            # Apply limit if specified
            if limit_per_year:
                cve_files = cve_files[:limit_per_year]
            
            # Load CVE files with progress bar
            for cve_file in tqdm(cve_files, desc=f"Loading {year}", leave=False):
                try:
                    with open(cve_file, 'r', encoding='utf-8') as f:
                        cve_data = json.load(f)
                        year_cves.append(cve_data)
                except Exception as e:
                    # Skip problematic files
                    continue
            
            cves.extend(year_cves)
            print(f"  Year {year}: Loaded {len(year_cves)} CVEs")
        
        return cves
    
    def parse_cve(self, cve_json: Dict) -> Optional[Dict]:
        """
        Parse CVE JSON to extract relevant information
        
        Args:
            cve_json: CVE data in JSON format
            
        Returns:
            Parsed CVE dictionary
        """
        try:
            cve_id = cve_json.get('cveMetadata', {}).get('cveId', '')
            
            # Extract English descriptions
            descriptions = []
            containers = cve_json.get('containers', {}).get('cna', {})
            for desc in containers.get('descriptions', []):
                if desc.get('lang', '') == 'en':
                    descriptions.append(desc.get('value', ''))
            
            description = ' '.join(descriptions)
            
            # Extract affected products and versions
            affected = containers.get('affected', [])
            products = []
            for item in affected:
                vendor = item.get('vendor', '')
                product = item.get('product', '')
                versions = item.get('versions', [])
                version_info = ' '.join([f"version {v.get('version', '')}" for v in versions[:3]])
                products.append(f"{vendor} {product} {version_info}")
            
            # Extract references (URLs often contain useful context)
            references = containers.get('references', [])
            ref_tags = ' '.join([' '.join(ref.get('tags', [])) for ref in references[:5]])
            
            # Extract problem types (CWE information)
            problem_types = containers.get('problemTypes', [])
            cwe_info = []
            for pt in problem_types:
                for desc in pt.get('descriptions', []):
                    cwe_info.append(desc.get('description', ''))
            
            # Combine all text features
            combined_text = f"{description} Products: {' '.join(products)} Tags: {ref_tags} Weaknesses: {' '.join(cwe_info)}"
            
            # Extract CVSS v3 metrics
            metrics = containers.get('metrics', [])
            cvss_data = None
            
            for metric in metrics:
                if 'cvssV3_1' in metric:
                    cvss_data = metric['cvssV3_1']
                    break
                elif 'cvssV3_0' in metric:
                    cvss_data = metric['cvssV3_0']
                    break
            
            if not cvss_data or not description:
                return None
            
            parsed = {
                'cve_id': cve_id,
                'description': combined_text[:2048],  # Limit length
                'attackVector': cvss_data.get('attackVector', ''),
                'attackComplexity': cvss_data.get('attackComplexity', ''),
                'privilegesRequired': cvss_data.get('privilegesRequired', ''),
                'userInteraction': cvss_data.get('userInteraction', ''),
                'scope': cvss_data.get('scope', ''),
                'confidentialityImpact': cvss_data.get('confidentialityImpact', ''),
                'integrityImpact': cvss_data.get('integrityImpact', ''),
                'availabilityImpact': cvss_data.get('availabilityImpact', ''),
                'baseScore': cvss_data.get('baseScore', 0),
                'baseSeverity': cvss_data.get('baseSeverity', '')
            }
            
            # Validate all metrics are present and valid
            for metric, valid_values in self.config.cvss_metrics.items():
                if parsed.get(metric) not in valid_values:
                    return None
            
            return parsed
            
        except Exception as e:
            return None
    
    def load_and_prepare_data(self, data_path: str = './cves', 
                            years: List[int] = None,
                            limit_per_year: int = None,
                            require_cvss_v3: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and prepare CVE data for training from local repository
        
        Args:
            data_path: Path to cloned cvelistV5 repository
            years: Specific years to load (None for all available)
            limit_per_year: Maximum CVEs per year (None for all)
            require_cvss_v3: Only include CVEs with CVSS v3 scores
            
        Returns:
            DataFrame and encoded labels
        """
        # Load CVEs from local repository
        print("Loading CVE data from local repository...")
        all_cves = self.load_local_cves(data_path, years, limit_per_year)
        
        if not all_cves:
            print("No CVEs loaded. Please check the data path.")
            return pd.DataFrame(), {}
        
        print(f"\nTotal CVEs loaded: {len(all_cves)}")
        print("Parsing CVE data and extracting CVSS scores...")
        
        parsed_data = []
        failed_parse = 0
        no_cvss_v3 = 0
        invalid_metrics = 0
        year_distribution = {}
        
        for cve in tqdm(all_cves, desc="Parsing CVEs"):
            parsed = self.parse_cve(cve)
            if parsed:
                # Track year distribution
                cve_year = int(parsed['cve_id'].split('-')[1]) if parsed['cve_id'] else 0
                year_distribution[cve_year] = year_distribution.get(cve_year, 0) + 1
                parsed_data.append(parsed)
            else:
                # Track why parsing failed
                if cve:
                    containers = cve.get('containers', {}).get('cna', {})
                    metrics = containers.get('metrics', [])
                    has_cvss_v3 = any('cvssV3' in str(m) for m in metrics)
                    if not has_cvss_v3:
                        no_cvss_v3 += 1
                    else:
                        invalid_metrics += 1
                else:
                    failed_parse += 1
        
        df = pd.DataFrame(parsed_data)
        
        print(f"\nParsing Results:")
        print(f"  Successfully parsed: {len(df)} CVEs with CVSS v3")
        print(f"  No CVSS v3 scores: {no_cvss_v3}")
        print(f"  Invalid/incomplete metrics: {invalid_metrics}")
        print(f"  Failed to parse: {failed_parse}")
        
        if len(df) == 0:
            print("\nNo valid CVEs with CVSS v3 scores found!")
            return df, {}
        
        # Show year distribution of CVEs with CVSS v3
        print(f"\nYear Distribution (CVEs with CVSS v3):")
        sorted_years = sorted(year_distribution.items())
        for year, count in sorted_years[-10:]:  # Show last 10 years
            print(f"  {year}: {count:5d} CVEs")
        
        # Show CVE ID range
        print(f"\nCVE ID Range:")
        print(f"  Earliest: {df['cve_id'].min()}")
        print(f"  Latest: {df['cve_id'].max()}")
        
        # Show distribution of CVSS metrics
        print(f"\nCVSS Metrics Distribution:")
        for metric in self.config.cvss_metrics.keys():
            if metric in df.columns:
                dist = df[metric].value_counts()
                print(f"\n  {metric}:")
                for value, count in dist.items():
                    print(f"    {value:20s}: {count:5d} ({count/len(df)*100:.1f}%)")
        
        # Show severity distribution
        severity_dist = df['baseSeverity'].value_counts()
        print(f"\nSeverity Distribution:")
        for severity, count in severity_dist.items():
            print(f"  {severity:8s}: {count:5d} ({count/len(df)*100:.1f}%)")
        
        # Show base score statistics
        print(f"\nBase Score Statistics:")
        print(f"  Mean: {df['baseScore'].mean():.2f}")
        print(f"  Median: {df['baseScore'].median():.2f}")
        print(f"  Std Dev: {df['baseScore'].std():.2f}")
        print(f"  Min: {df['baseScore'].min():.1f}")
        print(f"  Max: {df['baseScore'].max():.1f}")
        
        # Encode labels
        labels = {}
        for metric in self.config.cvss_metrics.keys():
            labels[metric] = self.label_encoders[metric].transform(df[metric].values)
        
        return df, labels
    
    def train(self, train_texts: List[str], train_labels: Dict,
             val_texts: List[str] = None, val_labels: Dict = None):
        """
        Enhanced training with advanced optimization techniques
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
        """
        print(f"Training on {len(train_texts)} samples with enhanced techniques")
        print(f"Using device: {self.device}")
        
        # Apply text augmentation if configured
        if self.config.use_augmentation:
            print("Applying text augmentation...")
            augmented_texts = []
            augmented_labels = {k: [] for k in train_labels.keys()}
            
            for i in tqdm(range(len(train_texts)), desc="Augmenting"):
                if np.random.random() < self.config.augmentation_prob:
                    aug_text = self.augment_text(train_texts[i])
                    augmented_texts.append(aug_text)
                    for metric in train_labels.keys():
                        augmented_labels[metric].append(train_labels[metric][i])
            
            # Add augmented data to training set
            train_texts = train_texts + augmented_texts
            for metric in train_labels.keys():
                train_labels[metric] = np.concatenate([
                    train_labels[metric],
                    np.array(augmented_labels[metric])
                ])
            
            print(f"Training set expanded to {len(train_texts)} samples with augmentation")
        
        # Compute class weights if configured
        class_weights = None
        if self.config.use_class_weights:
            print("Computing class weights for balanced training...")
            class_weights = self.compute_class_weights(train_labels)
        
        # Create datasets
        train_dataset = CVEDataset(
            train_texts, train_labels,
            self.tokenizer, self.config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2 if self.device.type == 'cuda' else 0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_texts and val_labels:
            val_dataset = CVEDataset(
                val_texts, val_labels,
                self.tokenizer, self.config.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size * 2,  # Larger batch for validation
                shuffle=False,
                num_workers=2 if self.device.type == 'cuda' else 0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Setup optimizer with different learning rates for different layers
        optimizer_params = [
            {'params': self.model.bert.embeddings.parameters(), 'lr': self.config.learning_rate * 0.1},
            {'params': self.model.bert.encoder.parameters(), 'lr': self.config.learning_rate * 0.5},
            {'params': self.model.feature_layer.parameters(), 'lr': self.config.learning_rate},
            {'params': self.model.classifiers.parameters(), 'lr': self.config.learning_rate * 2}
        ]
        
        optimizer = AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss functions
        criterion = {}
        for metric in self.config.cvss_metrics.keys():
            if self.config.use_focal_loss:
                criterion[metric] = FocalLoss(
                    alpha=self.config.focal_alpha,
                    gamma=self.config.focal_gamma
                )
            else:
                weight = class_weights[metric].to(self.device) if class_weights else None
                criterion[metric] = nn.CrossEntropyLoss(
                    weight=weight,
                    label_smoothing=self.config.label_smoothing
                )
        
        # Auxiliary severity loss
        criterion['severity'] = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        
        # Training loop with early stopping
        best_val_accuracy = 0
        patience_counter = 0
        best_model_state = None
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Unfreeze embeddings after specified epoch
            if self.config.freeze_embeddings and epoch >= self.config.unfreeze_after_epoch:
                self.model.unfreeze_embeddings()
                print("Unfreezing BERT embeddings for fine-tuning")
            
            # Training phase
            train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")
            
            accumulated_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_pbar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss for each metric
                total_loss = 0
                metric_losses = {}
                
                for metric in self.config.cvss_metrics.keys():
                    loss = criterion[metric](outputs[metric], labels[metric])
                    metric_losses[metric] = loss.item()
                    total_loss += loss
                
                # Add auxiliary severity prediction loss (with lower weight)
                if 'severity' in outputs:
                    # Derive severity from base metrics
                    severity_labels = self._derive_severity_labels(labels)
                    if severity_labels is not None:
                        severity_loss = criterion['severity'](outputs['severity'], severity_labels)
                        total_loss += severity_loss * 0.1  # Lower weight for auxiliary task
                
                # Gradient accumulation
                total_loss = total_loss / self.config.gradient_accumulation_steps
                total_loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += total_loss.item() * self.config.gradient_accumulation_steps
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{total_loss.item() * self.config.gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Calculate average validation accuracy
                avg_val_accuracy = np.mean(list(val_metrics.values()))
                
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Avg Val Accuracy: {avg_val_accuracy:.3f}")
                
                # Detailed metrics
                print("\nValidation Accuracies by Metric:")
                for metric, acc in val_metrics.items():
                    print(f"  {metric:25s}: {acc:.3f}")
                    if metric not in self.history['val_accuracy']:
                        self.history['val_accuracy'][metric] = []
                    self.history['val_accuracy'][metric].append(acc)
                
                # Early stopping check
                if avg_val_accuracy > best_val_accuracy:
                    best_val_accuracy = avg_val_accuracy
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    print(f"✓ New best model! Average accuracy: {best_val_accuracy:.3f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
                
                # Reduce learning rate on plateau
                if patience_counter >= self.config.reduce_lr_patience:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.5, self.config.min_lr)
                    print(f"Reducing learning rate")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model with accuracy: {best_val_accuracy:.3f}")
    
    def _derive_severity_labels(self, labels: Dict) -> Optional[torch.Tensor]:
        """
        Derive severity labels from CVSS metrics for auxiliary task
        
        Args:
            labels: Dictionary of metric labels
            
        Returns:
            Severity labels tensor
        """
        try:
            batch_size = labels[list(labels.keys())[0]].size(0)
            severity_labels = []
            
            for i in range(batch_size):
                # Simple heuristic for severity
                high_impact = sum([
                    labels['confidentialityImpact'][i] == 2,  # HIGH
                    labels['integrityImpact'][i] == 2,  # HIGH
                    labels['availabilityImpact'][i] == 2  # HIGH
                ])
                
                network_vector = labels['attackVector'][i] == 0  # NETWORK
                no_priv = labels['privilegesRequired'][i] == 0  # NONE
                
                if high_impact >= 2 and network_vector and no_priv:
                    severity = 4  # CRITICAL
                elif high_impact >= 2:
                    severity = 3  # HIGH
                elif high_impact >= 1:
                    severity = 2  # MEDIUM
                elif high_impact == 0 and network_vector:
                    severity = 1  # LOW
                else:
                    severity = 0  # NONE
                
                severity_labels.append(severity)
            
            return torch.tensor(severity_labels, dtype=torch.long).to(self.device)
        except:
            return None
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Evaluate the model
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Average loss and accuracy metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = {metric: [] for metric in self.config.cvss_metrics.keys()}
        true_labels = {metric: [] for metric in self.config.cvss_metrics.keys()}
        
        criterion = {}
        for metric in self.config.cvss_metrics.keys():
            criterion[metric] = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                batch_loss = 0
                for metric in self.config.cvss_metrics.keys():
                    loss = criterion[metric](outputs[metric], labels[metric])
                    batch_loss += loss
                    
                    # Store predictions
                    preds = torch.argmax(outputs[metric], dim=1)
                    predictions[metric].extend(preds.cpu().numpy())
                    true_labels[metric].extend(labels[metric].cpu().numpy())
                
                total_loss += batch_loss.item()
        
        # Calculate accuracies
        accuracies = {}
        for metric in self.config.cvss_metrics.keys():
            acc = accuracy_score(true_labels[metric], predictions[metric])
            accuracies[metric] = acc
        
        avg_loss = total_loss / len(data_loader)
        self.model.train()
        
        return avg_loss, accuracies
    
    def predict(self, text: str) -> Dict:
        """
        Predict CVSS scores for a new CVE description
        
        Args:
            text: CVE description
            
        Returns:
            Dictionary with predicted CVSS metrics
        """
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        predictions = {}
        
        for metric in self.config.cvss_metrics.keys():
            # Get prediction
            logits = outputs[metric]
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
            
            # Decode label
            pred_label = self.label_encoders[metric].inverse_transform([pred_idx])[0]
            predictions[metric] = pred_label
            
            # Get confidence
            confidence = probs[0, pred_idx].item()
            predictions[f'{metric}_confidence'] = confidence
        
        # Calculate base score
        base_score = self.calculate_cvss_score(predictions)
        predictions['baseScore'] = base_score
        predictions['baseSeverity'] = self.get_severity(base_score)
        
        return predictions
    
    def calculate_cvss_score(self, metrics: Dict) -> float:
        """
        Calculate CVSS v3.1 base score
        
        Args:
            metrics: Dictionary with CVSS metrics
            
        Returns:
            CVSS base score
        """
        # CVSS v3.1 scoring weights
        weights = {
            'attackVector': {'NETWORK': 0.85, 'ADJACENT_NETWORK': 0.62, 'LOCAL': 0.55, 'PHYSICAL': 0.2},
            'attackComplexity': {'LOW': 0.77, 'HIGH': 0.44},
            'privilegesRequired': {
                'NONE': 0.85,
                'LOW': 0.62 if metrics.get('scope') == 'UNCHANGED' else 0.68,
                'HIGH': 0.27 if metrics.get('scope') == 'UNCHANGED' else 0.5
            },
            'userInteraction': {'NONE': 0.85, 'REQUIRED': 0.62},
            'confidentialityImpact': {'HIGH': 0.56, 'LOW': 0.22, 'NONE': 0},
            'integrityImpact': {'HIGH': 0.56, 'LOW': 0.22, 'NONE': 0},
            'availabilityImpact': {'HIGH': 0.56, 'LOW': 0.22, 'NONE': 0}
        }
        
        # Calculate ISS (Impact Sub Score)
        c_impact = weights['confidentialityImpact'][metrics.get('confidentialityImpact', 'NONE')]
        i_impact = weights['integrityImpact'][metrics.get('integrityImpact', 'NONE')]
        a_impact = weights['availabilityImpact'][metrics.get('availabilityImpact', 'NONE')]
        
        iss = 1 - ((1 - c_impact) * (1 - i_impact) * (1 - a_impact))
        
        # Calculate Impact
        if metrics.get('scope') == 'UNCHANGED':
            impact = 6.42 * iss
        else:
            impact = 7.52 * (iss - 0.029) - 3.25 * pow(iss - 0.02, 15)
        
        # Calculate Exploitability
        av = weights['attackVector'][metrics.get('attackVector', 'NETWORK')]
        ac = weights['attackComplexity'][metrics.get('attackComplexity', 'LOW')]
        pr = weights['privilegesRequired'][metrics.get('privilegesRequired', 'NONE')]
        ui = weights['userInteraction'][metrics.get('userInteraction', 'NONE')]
        
        exploitability = 8.22 * av * ac * pr * ui
        
        # Calculate Base Score
        if impact <= 0:
            return 0.0
        
        if metrics.get('scope') == 'UNCHANGED':
            base_score = min(impact + exploitability, 10)
        else:
            base_score = min(1.08 * (impact + exploitability), 10)
        
        # Round to one decimal place
        return round(base_score, 1)
    
    def get_severity(self, score: float) -> str:
        """
        Get severity rating from base score
        
        Args:
            score: CVSS base score
            
        Returns:
            Severity rating
        """
        if score == 0:
            return 'NONE'
        elif score < 4.0:
            return 'LOW'
        elif score < 7.0:
            return 'MEDIUM'
        elif score < 9.0:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label_encoders': self.label_encoders,
            'history': self.history
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.label_encoders = checkpoint['label_encoders']
        self.history = checkpoint.get('history', {})
        
        self.model = CVSSTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")


def main():
    model_path = 'cvss_transformer_advanced_model.pt'
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        user_input = input("Do you want to:\n1. Load existing model (no training)\n2. Retrain from scratch\nChoice (1/2): ")
        
        if user_input == "1":
            # Load and use existing model
            config = CVSSConfig(
                model_name="microsoft/codebert-base",
                max_length=256
            )
            predictor = CVSSPredictor(config)
            predictor.load_model(model_path)
            
            # Skip to testing/prediction part
            print("Model loaded! Ready for predictions.")
            
            # Test on some examples
            test_cve = "A remote code execution vulnerability..."
            result = predictor.predict(test_cve)
            print(f"Prediction: {result}")
            return
    
    # Otherwise continue with normal training...
    print("Starting training from scratch...")
    # ... rest of the training code

    """
    Main function with advanced training configuration for maximum performance
    """
    print("=" * 70)
    print("CVE CVSS Transformer Prediction System - ADVANCED")
    print("Using Local CVE Repository with Performance Optimizations")
    print("=" * 70)
    
    # Advanced configuration for maximum performance
    config = CVSSConfig(
        # Use microsoft/codebert-base as default (more stable)
        # You can try other models if they work in your environment:
        model_name="microsoft/codebert-base",  # Stable and security-focused
        # Alternative models to try:
        # "bert-base-uncased" - standard BERT, very stable
        # "roberta-base" - often better than BERT
        # "jackaduma/SecBERT" - security-focused (if available)
        # "microsoft/deberta-v3-base" - best performance but may have compatibility issues
        
        # Training hyperparameters
        batch_size=8 if torch.cuda.is_available() else 8,
        gradient_accumulation_steps=4,  # Effective batch size = 64
        num_epochs=20,  # More epochs with early stopping
        learning_rate=1e-5,  # Lower LR for larger models
        warmup_ratio=0.1,
        weight_decay=0.01,
        dropout_rate=0.15,
        
        # Advanced techniques - all enabled for best performance
        use_class_weights=True,  # Handle imbalanced classes
        use_focal_loss=True,  # Better for hard examples
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_mixup=True,  # Data augmentation
        mixup_alpha=0.2,
        label_smoothing=0.1,  # Prevent overconfidence
        
        # Architecture enhancements
        use_lstm=True,  # Add sequence modeling
        lstm_hidden_size=256,
        num_lstm_layers=2,
        freeze_embeddings=True,  # Freeze initially for stability
        unfreeze_after_epoch=3,
        
        # Data augmentation
        use_augmentation=True,
        augmentation_prob=0.3,
        
        # Training optimization
        early_stopping_patience=5,
        reduce_lr_patience=3,
        min_lr=1e-7,
        
        max_length=512  # Can increase to 768 for longer descriptions
    )
    
    print(f"\nConfiguration Summary:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Advanced Features:")
    print(f"    - Focal Loss: {config.use_focal_loss}")
    print(f"    - Class Weighting: {config.use_class_weights}")
    print(f"    - LSTM Layers: {config.use_lstm}")
    print(f"    - Data Augmentation: {config.use_augmentation}")
    print(f"    - Label Smoothing: {config.label_smoothing}")
    print(f"    - Mixup: {config.use_mixup}")
    
    # Initialize predictor with error handling
    try:
        predictor = CVSSPredictor(config)
    except Exception as e:
        print(f"\nError loading model {config.model_name}: {e}")
        print("\nFalling back to bert-base-uncased...")
        config.model_name = "bert-base-uncased"
        predictor = CVSSPredictor(config)
    
    # Load data from local repository
    print("\n" + "=" * 70)
    print("Loading CVE Data from Local Repository")
    print("=" * 70)
    
    CVE_DATA_PATH = "./cve-data/cves"
    
    # Load ALL CVEs with CVSS v3 for maximum training data
    df, labels = predictor.load_and_prepare_data(
        data_path=CVE_DATA_PATH,
        years=None,  # All years - captures retroactively scored CVEs
        limit_per_year=None,  # No limit - use all available data
        require_cvss_v3=True
    )
    
    if len(df) < 100:
        print("\n⚠️  Insufficient data. Loading with specific years...")
        # Fallback to specific years if full load fails
        df, labels = predictor.load_and_prepare_data(
            data_path=CVE_DATA_PATH,
            years=list(range(1999, 2025)),
            limit_per_year=1000000,  # Limit per year for memory
            require_cvss_v3=True
        )
    
    if len(df) < 50:
        print("\n⚠️  Still insufficient data. Please check your CVE repository.")
        return
    
    print(f"\n✓ Loaded {len(df)} CVEs with complete CVSS v3 scores")
    
    # Advanced data splitting strategy
    texts = df['description'].tolist()
    
    # Stratified split to maintain class distribution
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Use severity as stratification target
    stratify_target = df['baseSeverity'].values

    # CUSTOMIZABLE SPLIT RATIOS - Adjust these values
    # Example: 85% train+val, 5% test (more training data)
    TEST_SIZE = 0.05  # 5% for test (was 0.10)
    VAL_SIZE = 0.056   # 10% of remaining for validation (was 0.15)

    # First split: separate test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_val_idx, test_idx = next(sss.split(texts, stratify_target))
    
    # Second split: separate validation from training
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=42)
    train_idx, val_idx = next(sss_val.split(
        [texts[i] for i in train_val_idx],
        stratify_target[train_val_idx]
    ))
    
    # Map back to original indices
    train_idx = [train_val_idx[i] for i in train_idx]
    val_idx = [train_val_idx[i] for i in val_idx]
    
    print(f"\nStratified Data Split:")
    print(f"  Training samples: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    print(f"  Validation samples: {len(val_idx)} ({len(val_idx)/len(df)*100:.1f}%)")
    print(f"  Test samples: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")
    
    # Verify distribution is maintained
    print(f"\nSeverity Distribution Check:")
    train_severities = df.iloc[train_idx]['baseSeverity'].value_counts(normalize=True)
    print(f"  Training: {dict(train_severities.head())}")
    val_severities = df.iloc[val_idx]['baseSeverity'].value_counts(normalize=True)
    print(f"  Validation: {dict(val_severities.head())}")
    
    # Prepare datasets
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]
    
    train_labels = {}
    val_labels = {}
    test_labels = {}
    
    for metric in labels.keys():
        train_labels[metric] = labels[metric][train_idx]
        val_labels[metric] = labels[metric][val_idx]
        test_labels[metric] = labels[metric][test_idx]
    
    # Train the model
    print("\n" + "=" * 70)
    print("Training Enhanced Transformer Model")
    print("=" * 70)
    
    predictor.train(
        train_texts, train_labels,
        val_texts, val_labels
    )
    
    # Save the model
#    model_path = 'cvss_transformer_advanced_model.pt'
    predictor.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Comprehensive evaluation on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    test_dataset = CVEDataset(
        test_texts, test_labels,
        predictor.tokenizer, config.max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False
    )
    
    test_loss, test_metrics = predictor.evaluate(test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"  Average Loss: {test_loss:.4f}")
    print(f"\n  Accuracies by Metric:")
    
    metric_groups = {
        'Exploitability': ['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteraction'],
        'Impact': ['scope', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact']
    }
    
    for group_name, group_metrics in metric_groups.items():
        print(f"\n  {group_name} Metrics:")
        group_accs = []
        for metric in group_metrics:
            if metric in test_metrics:
                acc = test_metrics[metric]
                group_accs.append(acc)
                print(f"    {metric:25s}: {acc:.3f}")
        if group_accs:
            print(f"    {group_name} Average: {np.mean(group_accs):.3f}")
    
    overall_accuracy = np.mean(list(test_metrics.values()))
    print(f"\n  Overall Average Accuracy: {overall_accuracy:.3f}")
    
    # Performance breakdown by severity
    print("\n" + "=" * 70)
    print("Performance Analysis by Severity")
    print("=" * 70)
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Predict on test set and analyze by severity
    test_df = df.iloc[test_idx].copy()
    predictions = []
    
    print("\nGenerating predictions for test set...")
    for text in tqdm(test_texts[:100], desc="Predicting"):  # Sample for analysis
        pred = predictor.predict(text)
        predictions.append(pred)
    
    # Analyze prediction accuracy by original severity
    severity_performance = {}
    for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        severity_mask = test_df['baseSeverity'].values[:100] == severity
        if severity_mask.any():
            severity_preds = [p for i, p in enumerate(predictions) if severity_mask[i]]
            if severity_preds:
                avg_score_diff = np.mean([
                    abs(p['baseScore'] - test_df.iloc[i]['baseScore'])
                    for i, p in enumerate(predictions) if severity_mask[i]
                ])
                severity_performance[severity] = avg_score_diff
    
    print("\nAverage Base Score Error by Severity:")
    for severity, error in severity_performance.items():
        print(f"  {severity:8s}: ±{error:.2f} points")
    
    # Test on example CVEs
    print("\n" + "=" * 70)
    print("Testing on Example CVE Descriptions")
    print("=" * 70)
    
    test_cases = [
        {
            "description": "A heap-based buffer overflow vulnerability in the TLS packet parsing functionality of NetworkDevice firmware versions 1.0 through 3.5 allows remote unauthenticated attackers to execute arbitrary code with root privileges or cause a denial of service via specially crafted TLS handshake messages.",
            "expected": "CRITICAL severity, Network vector, No privileges"
        },
        {
            "description": "An improper input validation vulnerability in the PDF rendering engine allows remote attackers to read arbitrary files on the system when a user opens a malicious PDF document. The vulnerability requires user interaction to exploit.",
            "expected": "MEDIUM-HIGH severity, Network vector, User interaction required"
        },
        {
            "description": "A race condition in the kernel's memory management subsystem could allow a local user with low privileges to escalate privileges to root. Successful exploitation requires precise timing and is considered difficult to achieve reliably.",
            "expected": "HIGH severity, Local vector, High complexity"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test Case {i}:")
        print(f"Description: {test_case['description'][:150]}...")
        print(f"Expected: {test_case['expected']}")
        
        predictions = predictor.predict(test_case['description'])
        
        print(f"\nPredicted CVSS Metrics:")
        print(f"  Base Score: {predictions['baseScore']} ({predictions['baseSeverity']})")
        
        # Show high-confidence predictions
        print(f"\n  High Confidence Predictions (>80%):")
        for metric in config.cvss_metrics.keys():
            confidence = predictions.get(f'{metric}_confidence', 0)
            if confidence > 0.8:
                print(f"    {metric}: {predictions[metric]} ({confidence:.1%})")
        
        # Show low-confidence predictions that might need review
        print(f"\n  Low Confidence Predictions (<60%):")
        for metric in config.cvss_metrics.keys():
            confidence = predictions.get(f'{metric}_confidence', 0)
            if confidence < 0.6:
                print(f"    {metric}: {predictions[metric]} ({confidence:.1%}) ⚠️")
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("Training Complete - Performance Summary")
    print("=" * 70)
    
    print(f"\nModel Statistics:")
    print(f"  Total Training Samples: {len(train_texts)}")
    if config.use_augmentation:
        print(f"  With Augmentation: ~{int(len(train_texts) * (1 + config.augmentation_prob))}")
    print(f"  Model Parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
    print(f"  Final Test Accuracy: {overall_accuracy:.3f}")
    
    print(f"\nRecommendations for Further Improvement:")
    print("  1. Ensemble multiple models (CodeBERT + RoBERTa + BERT)")
    print("  2. Use k-fold cross-validation for more robust training")
    print("  3. Fine-tune on domain-specific CVEs if targeting specific software")
    print("  4. Implement active learning to label ambiguous cases")
    print("  5. Add CWE category as additional input feature")
    print("  6. Use contrastive learning for better metric relationships")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. For production use, train with all available years")
    print("  2. Increase epochs to 15-20 for better accuracy")
    print("  3. Consider fine-tuning on your specific vulnerability domain")
    print("  4. Use the saved model for batch predictions on new CVEs")



# Add a utility function for batch prediction
def predict_batch_cves(model_path: str, cve_descriptions: List[str]) -> pd.DataFrame:
    """
    Predict CVSS scores for multiple CVE descriptions
    
    Args:
        model_path: Path to saved model
        cve_descriptions: List of CVE descriptions
        
    Returns:
        DataFrame with predictions
    """
    # Load model
    config = CVSSConfig()
    predictor = CVSSPredictor(config)
    predictor.load_model(model_path)
    
    # Make predictions
    results = []
    for desc in tqdm(cve_descriptions, desc="Predicting"):
        pred = predictor.predict(desc)
        pred['description'] = desc[:100] + '...' if len(desc) > 100 else desc
        results.append(pred)
    
    return pd.DataFrame(results)

if __name__=="__main__":
    main()
