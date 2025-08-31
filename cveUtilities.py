
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json
from dataclasses import dataclass

@dataclass
class cveInfo:
    id: str=''
    description: str=''
    av: str=''
    c: bool = True
    i: bool = True
    a: bool = True

def get_leaf_values(data):
    """
    Recursively extracts all leaf values from a nested JSON-like structure (dict or list).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            yield from get_leaf_values(value) # Yield from recursive call for dictionary values
    elif isinstance(data, list):
        for item in data:
            yield from get_leaf_values(item) # Yield from recursive call for list items
    else:
        yield data # If it's not a dict or list, it's a leaf value

def parse_CVSS(cvss):
    av = ''
    c = True
    i = True
    a = True
    for item in cvss.split('/'):
        if item.startswith('AV:'):
            if item.endswith("N"):
                av = 'Network'
            elif item.endswith("L"):
                av = "Local"
            elif item.endswith("P"):
                av = "Physical"
            elif item.endswith('A'):
                av = "Adjacent"
        if item.startswith("C:"):
            if item.endswith("N"):
                c = False 
        if item.startswith("I:"):
            if item.endswith("N"):
                i = False 
        if item.startswith("A:"):
            if item.endswith("N"):
                a = False 

    return (av,c,i,a)



def get_training_items(filelist):
    trainingItems = []
    with open(filelist,'r') as flist:
        for fpath in flist:
            item = cveInfo()
            item.id = fpath.split("/")[-1][:-6]
            with open(fpath.strip(),'r') as f:
                target = json.load(f)
                item.description = target["containers"]["cna"]["descriptions"][0]['value']

                values = get_leaf_values(target)
                cvss=[]
                for value in values:
                    if str(value).startswith("CVSS:"):
                        cvss.append(value)
                if len(cvss)>0:
                    (av,c,i,a) = parse_CVSS(cvss[0])
                    item.av = av
                    item.c = c
                    item.i = i
                    item.a = a
                if len(cvss) > 1:
                    if cvss[0] != cvss[1]:
                        print(f"WARNING: cvss1 does not match cvss0 for {item.id}")
            trainingItems.append(item)
    return(trainingItems)

def get_training_items_full(filelist):
    trainingItems = []
    with open(filelist,'r') as flist:
        for fpath in flist:
            item = cveInfo()
            item.id = fpath.split("/")[-1][:-6]
            with open(fpath.strip(),'r') as f:
                target = json.load(f)
                #item.description = target["containers"]["cna"]["descriptions"][0]['value']

                values = get_leaf_values(target)
                cvss=[]
                for value in values:
                    if str(value).startswith("CVSS:"):
                        cvss.append(value)
                    else:
                        item.description += f"{value};"
                if len(cvss)>0:
                    (av,c,i,a) = parse_CVSS(cvss[0])
                    item.av = av
                    item.c = c
                    item.i = i
                    item.a = a
                if len(cvss) > 1:
                    if cvss[0] != cvss[1]:
                        print(f"WARNING: cvss1 does not match cvss0 for {item.id}")
            trainingItems.append(item)
    return(trainingItems)