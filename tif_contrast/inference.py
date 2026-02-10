import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import open_clip

from .models.tif_model import TIFContrast


class TIFInference:
    """Inference wrapper for TIF-Contrast model"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = TIFContrast.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load image transform
        _, _, self.transform = open_clip.create_model_and_transforms('ViT-B-32')
        
        print(f"Model loaded on {self.device}")
    
    def load_document(self, doc_path: str):
        doc = {}
        
        # Load text
        text_path = os.path.join(doc_path, 'text.txt')
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                doc['text'] = f.read()
        else:
            doc['text'] = ""
        
        # Load image
        image_path = os.path.join(doc_path, 'figure.png')
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            doc['image'] = self.transform(image).to(self.device)
        else:
            doc['image'] = None
        
        # Load formula
        formula_path = os.path.join(doc_path, 'formula.txt')
        if os.path.exists(formula_path):
            with open(formula_path, 'r', encoding='utf-8') as f:
                doc['formula'] = f.read()
        else:
            doc['formula'] = ""
        
        return doc
    
    def compute_similarity(self, doc1_path: str, doc2_path: str) -> float:
        # Load documents
        doc1 = self.load_document(doc1_path)
        doc2 = self.load_document(doc2_path)
        
        # Compute similarity
        with torch.no_grad():
            similarity = self.model.compute_similarity(doc1, doc2)
        
        return similarity
    
    def get_interpretability(self, doc_path: str):
        doc = self.load_document(doc_path)
        
        with torch.no_grad():
            weights = self.model.get_attention_weights(
                text=doc['text'],
                image=doc['image'],
                formula=doc['formula']
            )
        
        return weights


def batch_inference(document_pairs: str,
                   model_path: str,
                   batch_size: int = 16,
                   output_file: str = None,
                   device: str = 'cuda'):
    # Load pairs
    pairs_df = pd.read_csv(document_pairs)
    
    # Initialize inference engine
    engine = TIFInference(model_path, device)
    
    # Compute similarities
    similarities = []
    
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc='Computing similarities'):
        doc1_path = row['doc1_path']
        doc2_path = row['doc2_path']
        
        try:
            similarity = engine.compute_similarity(doc1_path, doc2_path)
            similarities.append(similarity)
        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            similarities.append(0.0)
    
    # Add to dataframe
    pairs_df['similarity'] = similarities
    pairs_df['prediction'] = (pairs_df['similarity'] >= 0.5).astype(int)
    
    # Save results
    if output_file:
        pairs_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return pairs_df


def compare_documents(doc1_path: str,
                     doc2_path: str,
                     model_path: str,
                     verbose: bool = True):

    engine = TIFInference(model_path)
    
    # Compute similarity
    similarity = engine.compute_similarity(doc1_path, doc2_path)
    
    # Get attention weights for both documents
    weights1 = engine.get_interpretability(doc1_path)
    weights2 = engine.get_interpretability(doc2_path)
    
    result = {
        'similarity': similarity,
        'prediction': 'Similar' if similarity >= 0.5 else 'Dissimilar',
        'doc1_attention': weights1,
        'doc2_attention': weights2
    }
    
    if verbose:
        print("="*50)
        print("DOCUMENT COMPARISON")
        print("="*50)
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Prediction: {result['prediction']}")
        print("\nDocument 1 Attention Weights:")
        for modality, weight in weights1.items():
            print(f"  {modality}: {weight:.4f}")
        print("\nDocument 2 Attention Weights:")
        for modality, weight in weights2.items():
            print(f"  {modality}: {weight:.4f}")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with TIF-Contrast')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'],
                       default='single', help='Inference mode')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--doc1', type=str, help='Path to document 1')
    parser.add_argument('--doc2', type=str, help='Path to document 2')
    parser.add_argument('--pairs_file', type=str,
                       help='CSV file with document pairs (for batch mode)')
    parser.add_argument('--output_file', type=str,
                       help='Output file for batch results')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.doc1 or not args.doc2:
            print("Error: --doc1 and --doc2 required for single mode")
            exit(1)
        
        compare_documents(args.doc1, args.doc2, args.model_path)
    
    elif args.mode == 'batch':
        if not args.pairs_file:
            print("Error: --pairs_file required for batch mode")
            exit(1)
        
        batch_inference(
            args.pairs_file,
            args.model_path,
            output_file=args.output_file,
            device=args.device
        )
