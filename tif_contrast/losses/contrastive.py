import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise-Contrastive Estimation) Loss
    Used for pairwise modal alignment
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor: torch.Tensor, 
                positive: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
       
        batch_size = anchor.size(0)
        
        # Compute similarity scores
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch_size]
        
        # Negative similarities
        neg_sim = torch.matmul(anchor, negatives.transpose(1, 2)) / self.temperature
        # [batch_size, num_negatives]
        
        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+num_negatives]
        labels = torch.zeros(batch_size, dtype=torch.long).to(anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class PairwiseAlignmentLoss(nn.Module):
  
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def compute_pairwise_loss(self, emb1: torch.Tensor, 
                             emb2: torch.Tensor) -> torch.Tensor:
       
        batch_size = emb1.size(0)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(emb1, emb2.T) / self.temperature
        # [batch_size, batch_size]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size).to(emb1.device)
        
        # Compute loss in both directions
        loss_1to2 = F.cross_entropy(similarity_matrix, labels)
        loss_2to1 = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_1to2 + loss_2to1) / 2
        
        return loss
    
    def forward(self, embeddings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
       
        loss_total = 0.0
        num_pairs = 0
        
        modalities = list(embeddings_dict.keys())
        
        # Compute loss for all pairs
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                
                if mod1 in embeddings_dict and mod2 in embeddings_dict:
                    loss_pair = self.compute_pairwise_loss(
                        embeddings_dict[mod1],
                        embeddings_dict[mod2]
                    )
                    loss_total += loss_pair
                    num_pairs += 1
        
        # Average over all pairs
        if num_pairs > 0:
            loss_total = loss_total / num_pairs
        
        return loss_total


class TripletAlignmentLoss(nn.Module):
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
       
        batch_size = fused_embeddings.size(0)
        
        # Normalize
        fused_embeddings = F.normalize(fused_embeddings, p=2, dim=-1)
        
        # Compute self-similarity matrix
        similarity_matrix = torch.matmul(fused_embeddings, fused_embeddings.T) / self.temperature
        
        # Diagonal elements are self-similarities (should be maximized)
        # Off-diagonal elements are cross-document similarities (should be minimized)
        labels = torch.arange(batch_size).to(fused_embeddings.device)
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class TIFContrastLoss(nn.Module):
    
    def __init__(self, 
                 lambda_pair: float = 0.6,
                 lambda_triplet: float = 0.4,
                 temperature: float = 0.07):
        super().__init__()
        
        self.lambda_pair = lambda_pair
        self.lambda_triplet = lambda_triplet
        
        self.pairwise_loss = PairwiseAlignmentLoss(temperature)
        self.triplet_loss = TripletAlignmentLoss(temperature)
        
    def forward(self, 
                modal_embeddings: Dict[str, torch.Tensor],
                fused_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Pairwise alignment loss
        loss_pair = self.pairwise_loss(modal_embeddings)
        
        # Triplet alignment loss
        loss_triplet = self.triplet_loss(fused_embeddings)
        
        # Total loss
        loss_total = self.lambda_pair * loss_pair + self.lambda_triplet * loss_triplet
        
        return {
            'total': loss_total,
            'pairwise': loss_pair,
            'triplet': loss_triplet
        }
