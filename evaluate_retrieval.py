import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tif_contrast.models.tif_model import TIFContrast
from tif_contrast.data.dataset import AcademicDocumentDataset, collate_fn
from tif_contrast.utils.metrics import compute_retrieval_metrics

@torch.no_grad()
def evaluate_retrieval(model, dataloader, device):
    model.eval()
    embeddings_list = []
    
    print("Encoding document gallery for retrieval...")
    for batch in tqdm(dataloader):
        outputs = model(
            text=batch['text'],
            image=batch['image'].to(device),
            formula=batch['formula']
        )
        embeddings_list.append(outputs['fused'].cpu().numpy())
    
    # Concatenate all embeddings to form the search gallery
    gallery = np.concatenate(embeddings_list, axis=0)
    
    # Compute cross-similarity matrix (Cosine Similarity)
    # Norm is handled within the model/fusion layers
    sim_matrix = gallery @ gallery.T
    
    # Evaluate Recall@K (Diagonal elements are ground truth matches)
    labels = np.eye(len(gallery))
    k_vals = [1, 5, 10, 50]
    total_recall = {f'recall@{k}': 0 for k in k_vals}
    
    for i in range(len(sim_matrix)):
        metrics = compute_retrieval_metrics(sim_matrix[i], labels[i], k_values=k_vals)
        for k_key in total_recall:
            total_recall[k_key] += metrics[k_key]
            
    # Compute mean recall across all queries
    for k_key in total_recall:
        print(f"{k_key.upper()}: {total_recall[k_key]/len(gallery)*100:.2f}%")

if __name__ == "__main__":
    # Script entry point for standalone evaluation
    pass