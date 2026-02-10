import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import Dict, Tuple, Optional
import open_clip


class AcademicDocumentDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 pairs_file: str,
                 transform=None,
                 max_text_length: int = 512):
       
        self.data_dir = data_dir
        self.pairs_df = pd.read_csv(pairs_file)
        self.max_text_length = max_text_length
        
        # Setup image transform
        if transform is None:
            _, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32')
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.pairs_df)
    
    def load_document(self, doc_id: str) -> Dict[str, any]:
        
        doc_path = os.path.join(self.data_dir, 'documents', doc_id)
        
        doc_data = {}
        
        # Load text
        text_path = os.path.join(doc_path, 'text.txt')
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                doc_data['text'] = f.read().strip()
        else:
            doc_data['text'] = ""
        
        # Load image
        image_path = os.path.join(doc_path, 'figure.png')
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            doc_data['image'] = self.transform(image)
        else:
            # Create dummy image if missing
            doc_data['image'] = torch.zeros(3, 224, 224)
        
        # Load formula
        formula_path = os.path.join(doc_path, 'formula.txt')
        if os.path.exists(formula_path):
            with open(formula_path, 'r', encoding='utf-8') as f:
                doc_data['formula'] = f.read().strip()
        else:
            doc_data['formula'] = ""
        
        return doc_data
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, int]:
       
        row = self.pairs_df.iloc[idx]
        
        doc1_id = row['doc1_id']
        doc2_id = row['doc2_id']
        label = int(row['label'])  # 1 for similar, 0 for dissimilar
        
        # Load documents
        doc1_data = self.load_document(doc1_id)
        doc2_data = self.load_document(doc2_id)
        
        return doc1_data, doc2_data, label


class TripletDataset(Dataset):
    
    def __init__(self,
                 data_dir: str,
                 triplets_file: str,
                 transform=None):
       
        self.data_dir = data_dir
        self.triplets_df = pd.read_csv(triplets_file)
        
        if transform is None:
            _, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32')
        else:
            self.transform = transform
        
        self.base_dataset = AcademicDocumentDataset(
            data_dir=data_dir,
            pairs_file=triplets_file,  # Reuse loading logic
            transform=transform
        )
    
    def __len__(self) -> int:
        return len(self.triplets_df)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        
        row = self.triplets_df.iloc[idx]
        
        anchor_data = self.base_dataset.load_document(row['anchor_id'])
        positive_data = self.base_dataset.load_document(row['positive_id'])
        negative_data = self.base_dataset.load_document(row['negative_id'])
        
        return anchor_data, positive_data, negative_data


def collate_fn(batch):
   
    doc1_batch = {
        'text': [],
        'image': [],
        'formula': []
    }
    doc2_batch = {
        'text': [],
        'image': [],
        'formula': []
    }
    labels = []
    
    for doc1, doc2, label in batch:
        doc1_batch['text'].append(doc1['text'])
        doc1_batch['image'].append(doc1['image'])
        doc1_batch['formula'].append(doc1['formula'])
        
        doc2_batch['text'].append(doc2['text'])
        doc2_batch['image'].append(doc2['image'])
        doc2_batch['formula'].append(doc2['formula'])
        
        labels.append(label)
    
    # Stack images
    doc1_batch['image'] = torch.stack(doc1_batch['image'])
    doc2_batch['image'] = torch.stack(doc2_batch['image'])
    
    labels = torch.tensor(labels)
    
    return doc1_batch, doc2_batch, labels
