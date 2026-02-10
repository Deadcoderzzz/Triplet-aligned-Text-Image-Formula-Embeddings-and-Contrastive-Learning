import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class AdaptiveAttentionFusion(nn.Module):

    def __init__(self, embedding_dim: int = 512, num_modalities: int = 3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        
        # Query network for computing attention weights
        self.query_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Key network for each modality
        self.key_nets = nn.ModuleDict({
            'text': nn.Linear(embedding_dim, embedding_dim),
            'image': nn.Linear(embedding_dim, embedding_dim),
            'formula': nn.Linear(embedding_dim, embedding_dim)
        })
        
        # Value transformation
        self.value_nets = nn.ModuleDict({
            'text': nn.Linear(embedding_dim, embedding_dim),
            'image': nn.Linear(embedding_dim, embedding_dim),
            'formula': nn.Linear(embedding_dim, embedding_dim)
        })
        
    def compute_attention_weights(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, float]:

        scores = {}
        
        for modality, emb in embeddings.items():
            # Compute attention score
            score = self.query_net(emb)
            scores[modality] = score
        
        # Stack and apply softmax
        modality_list = list(scores.keys())
        score_tensor = torch.stack([scores[m] for m in modality_list])
        weights = F.softmax(score_tensor, dim=0)
        
        # Convert back to dictionary
        weight_dict = {modality_list[i]: weights[i].item() 
                      for i in range(len(modality_list))}
        
        return weight_dict
    
    def forward(self, embeddings: Dict[str, torch.Tensor], 
                dropout_rate: float = 0.1) -> torch.Tensor:

        # Apply modality dropout during training
        if self.training and dropout_rate > 0:
            embeddings = self.apply_modality_dropout(embeddings, dropout_rate)
        
        # Compute attention weights
        attention_weights = self.compute_attention_weights(embeddings)
        
        # Transform embeddings
        fused_embedding = torch.zeros(self.embedding_dim).to(
            next(iter(embeddings.values())).device
        )
        
        for modality, emb in embeddings.items():
            # Apply key and value transformations
            key = self.key_nets[modality](emb)
            value = self.value_nets[modality](emb)
            
            # Weighted sum
            weight = attention_weights[modality]
            fused_embedding += weight * value
        
        # L2 normalization
        fused_embedding = F.normalize(fused_embedding, p=2, dim=-1)
        
        return fused_embedding
    
    def apply_modality_dropout(self, embeddings: Dict[str, torch.Tensor], 
                              dropout_rate: float) -> Dict[str, torch.Tensor]:

        dropped_embeddings = {}
        
        for modality, emb in embeddings.items():
            if torch.rand(1).item() > dropout_rate:
                dropped_embeddings[modality] = emb
        
        # Ensure at least one modality remains
        if len(dropped_embeddings) == 0:
            random_modality = list(embeddings.keys())[0]
            dropped_embeddings[random_modality] = embeddings[random_modality]
        
        return dropped_embeddings


class InterpretableAttention(nn.Module):

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.scale = embedding_dim ** -0.5
        
    def forward(self, query: torch.Tensor, 
                keys: torch.Tensor) -> torch.Tensor:

        Q = self.query_proj(query.unsqueeze(0))  # [1, embedding_dim]
        K = self.key_proj(keys)  # [num_components, embedding_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        return attention_weights.squeeze(0)
