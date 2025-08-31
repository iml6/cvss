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
    model_name: str = "microsoft/codebert-base"  # Good for security-related text
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
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


class CVSSTransformer(nn.Module):
    """
    Transformer model for CVSS prediction
    Multi-head architecture for predicting multiple CVSS metrics
    """
    
    def __init__(self, config: CVSSConfig):
        """
        Initialize the CVSS transformer model
        
        Args:
            config: Model configuration
        """
        super(CVSSTransformer, self).__init__()
        
        self.config = config
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Freeze early BERT layers for efficiency (optional)
        # for param in self.bert.embeddings.parameters():
        #     param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # Shared feature extraction layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Separate classification heads for each CVSS metric
        self.classifiers = nn.ModuleDict()
        for metric, classes in config.cvss_metrics.items():
            self.classifiers[metric] = nn.Linear(hidden_size // 2, len(classes))
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads=8, 
            dropout=config.dropout_rate,
            batch_first=True
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary of logits for each CVSS metric
        """
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token embedding with attention-weighted sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        sequence_output = outputs.last_hidden_state
        
        # Apply self-attention to the sequence
        attn_output, _ = self.attention(
            sequence_output, 
            sequence_output, 
            sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Combine CLS and attention output
        pooled_output = torch.mean(attn_output, dim=1) + cls_embedding
        pooled_output = self.dropout(pooled_output)
        
        # Extract shared features
        features = self.feature_layer(pooled_output)
        
        # Get predictions for each metric
        logits = {}
        for metric, classifier in self.classifiers.items():
            logits[metric] = classifier(features)
        
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
        Train the transformer model
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
        """
        print(f"Training on {len(train_texts)} samples")
        print(f"Using device: {self.device}")
        
        # Create datasets
        train_dataset = CVEDataset(
            train_texts, train_labels, 
            self.tokenizer, self.config.max_length
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_texts and val_labels:
            val_dataset = CVEDataset(
                val_texts, val_labels,
                self.tokenizer, self.config.max_length
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss functions for each metric
        criterion = {}
        for metric in self.config.cvss_metrics.keys():
            criterion[metric] = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")
            
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss for each metric
                total_loss = 0
                for metric in self.config.cvss_metrics.keys():
                    loss = criterion[metric](outputs[metric], labels[metric])
                    total_loss += loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += total_loss.item()
                train_pbar.set_postfix({'loss': total_loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader:
                val_loss, val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print("Validation Accuracies:")
                for metric, acc in val_metrics.items():
                    print(f"  {metric}: {acc:.3f}")
                    if metric not in self.history['val_accuracy']:
                        self.history['val_accuracy'][metric] = []
                    self.history['val_accuracy'][metric].append(acc)
    
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
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.label_encoders = checkpoint['label_encoders']
        self.history = checkpoint.get('history', {})
        
        self.model = CVSSTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to train and test the CVE CVSS prediction system
    """
    print("=" * 70)
    print("CVE CVSS Transformer Prediction System")
    print("Using Local CVE Repository")
    print("=" * 70)
    
    # Configuration
    config = CVSSConfig(
        model_name="microsoft/codebert-base",  # Or "bert-base-uncased" for general BERT or "jackaduma/SecBERT"
        batch_size=16 if torch.cuda.is_available() else 4,
        num_epochs=10,  # Increase for better accuracy
        learning_rate=2e-5,
        max_length=512
    )
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    
    # Initialize predictor
    predictor = CVSSPredictor(config)
    
    # Load and prepare data from local repository
    print("\n" + "=" * 70)
    print("Loading CVE Data from Local Repository")
    print("=" * 70)
    
    # Path to your cloned repository
    CVE_DATA_PATH = "./cves"  # Adjust this path if needed
    
    # Option 1: Load ALL CVEs with CVSS v3 scores (including older CVEs retroactively scored)
    # df, labels = predictor.load_and_prepare_data(
    #     data_path=CVE_DATA_PATH,
    #     years=None,  # Load all years
    #     limit_per_year=None,  # No limit
    #     require_cvss_v3=True  # Only CVEs with CVSS v3 (regardless of year)
    # )
    
    # Option 2: Load specific years for balanced training
    df, labels = predictor.load_and_prepare_data(
        data_path=CVE_DATA_PATH,
        years=[2000, 2005, 2010, 2015, 2020, 2021, 2022, 2023, 2024],  # Mix of old and new
        limit_per_year=1000,  # Limit for faster testing (remove for production)
        require_cvss_v3=True  # Only those with CVSS v3
    )
    
    # Option 3: Load ALL years to get maximum CVSS v3 data
    # df, labels = predictor.load_and_prepare_data(
    #     data_path=CVE_DATA_PATH,
    #     years=list(range(1999, 2025)),  # All available years
    #     limit_per_year=None,
    #     require_cvss_v3=True
    # )
    
    if len(df) < 50:
        print("\n⚠️  Insufficient data. Please check:")
        print(f"  1. CVE data exists at: {CVE_DATA_PATH}")
        print("  2. The path contains year directories (e.g., ./cves/2023/)")
        print("  3. CVEs have CVSS v3 scores")
        return
    
    print(f"\n✓ Loaded {len(df)} CVEs with complete CVSS v3 scores")
    
    # Split data
    texts = df['description'].tolist()
    
    # Create train/validation/test split
    train_val_idx, test_idx = train_test_split(
        range(len(texts)), 
        test_size=0.1,  # 10% for final testing
        random_state=42
    )
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,  # 15% of remaining for validation
        random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Validation samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    
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
    print("Training Transformer Model")
    print("=" * 70)
    
    predictor.train(
        train_texts, train_labels,
        val_texts, val_labels
    )
    
    # Save the model
    model_path = 'cvss_transformer_model.pt'
    predictor.save_model(model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    from torch.utils.data import DataLoader
    test_dataset = CVEDataset(
        test_texts, test_labels,
        predictor.tokenizer, config.max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    test_loss, test_metrics = predictor.evaluate(test_loader)
    
    print(f"\nTest Set Performance:")
    print(f"  Average Loss: {test_loss:.4f}")
    print(f"\n  Accuracies by Metric:")
    
    for metric, acc in test_metrics.items():
        print(f"    {metric:25s}: {acc:.3f}")
    
    avg_accuracy = np.mean(list(test_metrics.values()))
    print(f"\n  Overall Average Accuracy: {avg_accuracy:.3f}")
    
    # Test predictions on example CVEs
    print("\n" + "=" * 70)
    print("Testing Predictions on Example CVEs")
    print("=" * 70)
    
    test_cases = [
        {
            "description": "A SQL injection vulnerability in the user authentication module of WebApp v2.1 allows remote attackers to execute arbitrary SQL commands via specially crafted username parameter, potentially leading to unauthorized database access and data exfiltration.",
            "expected": "High severity, Network vector, Low complexity"
        },
        {
            "description": "A buffer overflow vulnerability in the local print spooler service allows a locally authenticated user with low privileges to execute arbitrary code with system privileges by sending specially crafted print jobs.",
            "expected": "High severity, Local vector, High privileges required"
        },
        {
            "description": "Cross-site scripting (XSS) vulnerability in the comment section of BlogSoftware 3.0 allows authenticated users to inject malicious JavaScript that executes in other users' browsers when viewing affected pages.",
            "expected": "Medium severity, Network vector, User interaction required"
        },
        {
            "description": "An information disclosure vulnerability in the debug logs of CloudService API exposes sensitive authentication tokens when verbose logging is enabled, which could allow unauthorized access to user accounts.",
            "expected": "Medium-High severity, Network vector, High confidentiality impact"
        },
        {
            "description": "A race condition in the kernel memory allocator allows local users to cause a denial of service (system crash) or possibly gain elevated privileges via concurrent memory allocation requests.",
            "expected": "High severity, Local vector, High complexity"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"Test Case {i}:")
        print(f"Description: {test_case['description'][:100]}...")
        print(f"Expected: {test_case['expected']}")
        
        predictions = predictor.predict(test_case['description'])
        
        print(f"\nPredicted CVSS Metrics:")
        print(f"  Base Score: {predictions['baseScore']} ({predictions['baseSeverity']})")
        
        # Group metrics for better readability
        print(f"\n  Exploitability Metrics:")
        for metric in ['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteraction']:
            confidence = predictions.get(f'{metric}_confidence', 0)
            print(f"    {metric:25s}: {predictions[metric]:20s} (conf: {confidence:.1%})")
        
        print(f"\n  Impact Metrics:")
        for metric in ['scope', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact']:
            confidence = predictions.get(f'{metric}_confidence', 0)
            print(f"    {metric:25s}: {predictions[metric]:20s} (conf: {confidence:.1%})")
    
    # Print training summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    print(f"\nModel Information:")
    print(f"  Model saved to: {model_path}")
    print(f"  Total parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in predictor.model.parameters() if p.requires_grad):,}")
    
    if predictor.history['val_accuracy']:
        print(f"\nFinal Validation Performance:")
        for metric, acc_history in predictor.history['val_accuracy'].items():
            if acc_history:
                improvement = (acc_history[-1] - acc_history[0]) * 100 if len(acc_history) > 1 else 0
                print(f"  {metric:25s}: {acc_history[-1]:.3f} ({improvement:+.1f}% improvement)")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. For production use, train with all available years")
    print("  2. Increase epochs to 15-20 for better accuracy")
    print("  3. Consider fine-tuning on your specific vulnerability domain")
    print("  4. Use the saved model for batch predictions on new CVEs")
    print("=" * 70)


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