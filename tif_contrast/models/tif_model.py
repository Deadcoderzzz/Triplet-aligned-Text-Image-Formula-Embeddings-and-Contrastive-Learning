import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .encoders import MultiModalEncoder
from .fusion import AdaptiveAttentionFusion, InterpretableAttention


class TIFContrast(nn.Module):
    """
    Triplet-aligned Text-Image-Formula Contrastive Similarity Framework
    """
    
    def __init__(self,
                 text_encoder_name: str = "scibert_scivocab_uncased",
                 image_encoder_name: str = "ViT-B-32",
                 formula_encoder_name: str = "MathBERT",
                 embedding_dim: int = 512,
                 use_adaptive_fusion: bool = True,
                 modality_dropout: float = 0.1):
        super().__init__()
        
        # Multi-modal encoders
        self.encoder = MultiModalEncoder(
            text_encoder_name=text_encoder_name,
            image_encoder_name=image_encoder_name,
            formula_encoder_name=formula_encoder_name,
            embedding_dim=embedding_dim
        )
        
        # Fusion module
        self.use_adaptive_fusion = use_adaptive_fusion
        if use_adaptive_fusion:
            self.fusion = AdaptiveAttentionFusion(embedding_dim)
        else:
            # Simple average fusion
            self.fusion = lambda x, **kwargs: torch.mean(
                torch.stack(list(x.values())), dim=0
            )
        
        self.modality_dropout = modality_dropout
        self.embedding_dim = embedding_dim
        
        # Interpretability module
        self.attention = InterpretableAttention(embedding_dim)
        
    def encode_document(self,
                       text: Optional[str] = None,
                       image: Optional[torch.Tensor] = None,
                       formula: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Encode a multi-modal document
        
        Args:
            text: Text content
            image: Image tensor
            formula: Formula LaTeX string
        Returns:
            (modal_embeddings_dict, fused_embedding)
        """
        # Extract modal embeddings
        modal_embeddings = self.encoder(text=text, image=image, formula=formula)
        
        # Fuse embeddings
        if self.use_adaptive_fusion:
            fused_embedding = self.fusion(
                modal_embeddings, 
                dropout_rate=self.modality_dropout if self.training else 0.0
            )
        else:
            fused_embedding = self.fusion(modal_embeddings)
        
        return modal_embeddings, fused_embedding
    
    def forward(self,
                doc1_text: Optional[str] = None,
                doc1_image: Optional[torch.Tensor] = None,
                doc1_formula: Optional[str] = None,
                doc2_text: Optional[str] = None,
                doc2_image: Optional[torch.Tensor] = None,
                doc2_formula: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with document pairs
        
        Returns:
            Dictionary with modal_embeddings and fused_embeddings for both documents
        """
        # Encode document 1
        doc1_modal_emb, doc1_fused = self.encode_document(
            text=doc1_text,
            image=doc1_image,
            formula=doc1_formula
        )
        
        # Encode document 2
        doc2_modal_emb, doc2_fused = self.encode_document(
            text=doc2_text,
            image=doc2_image,
            formula=doc2_formula
        )
        
        return {
            'doc1_modal': doc1_modal_emb,
            'doc1_fused': doc1_fused,
            'doc2_modal': doc2_modal_emb,
            'doc2_fused': doc2_fused
        }
    
    def compute_similarity(self,
                          doc1: Dict[str, any],
                          doc2: Dict[str, any]) -> float:
        """
        Compute similarity score between two documents
        
        Args:
            doc1: Dictionary with 'text', 'image', 'formula' keys
            doc2: Dictionary with 'text', 'image', 'formula' keys
        Returns:
            Similarity score in [0, 1]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode documents
            _, doc1_fused = self.encode_document(
                text=doc1.get('text'),
                image=doc1.get('image'),
                formula=doc1.get('formula')
            )
            
            _, doc2_fused = self.encode_document(
                text=doc2.get('text'),
                image=doc2.get('image'),
                formula=doc2.get('formula')
            )
            
            # Cosine similarity
            similarity = torch.cosine_similarity(
                doc1_fused.unsqueeze(0),
                doc2_fused.unsqueeze(0)
            ).item()
            
            # Convert to [0, 1] range
            similarity = (similarity + 1) / 2
        
        return similarity
    
    def get_attention_weights(self,
                             text: Optional[str] = None,
                             image: Optional[torch.Tensor] = None,
                             formula: Optional[str] = None) -> Dict[str, float]:
        """
        Get interpretable attention weights for each modality
        
        Returns:
            Dictionary with modality weights
        """
        self.eval()
        
        with torch.no_grad():
            modal_embeddings, _ = self.encode_document(
                text=text, image=image, formula=formula
            )
            
            if self.use_adaptive_fusion:
                weights = self.fusion.compute_attention_weights(modal_embeddings)
            else:
                # Equal weights for simple fusion
                num_modalities = len(modal_embeddings)
                weights = {k: 1.0/num_modalities for k in modal_embeddings.keys()}
        
        return weights
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract config
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        # Initialize model
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'use_adaptive_fusion': self.use_adaptive_fusion,
                'modality_dropout': self.modality_dropout
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
