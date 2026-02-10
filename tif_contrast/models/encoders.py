import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import open_clip
from typing import Dict, Optional


class TextEncoder(nn.Module):
    """SciBERT-based text encoder for scientific documents"""
    
    def __init__(self, model_name: str = "scibert_scivocab_uncased", 
                 embedding_dim: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Projection layer to unified embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Args:
            text: Input text string
            max_length: Maximum sequence length
        Returns:
            Normalized embedding vector [embedding_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(next(self.encoder.parameters()).device)
        
        # Extract features
        outputs = self.encoder(**inputs)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to unified space
        embeddings = self.projection(cls_output)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.squeeze(0)


class ImageEncoder(nn.Module):
    """CLIP-ViT + YOLOv11 based image encoder for scientific figures"""
    
    def __init__(self, clip_model: str = "ViT-B-32", 
                 pretrained: str = "laion2b_s34b_b79k",
                 embedding_dim: int = 512):
        super().__init__()
        
        # Load OpenCLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=pretrained
        )
        self.clip_dim = self.model.visual.output_dim
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Preprocessed image tensor [3, H, W]
        Returns:
            Normalized embedding vector [embedding_dim]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        
        # Project to unified space
        embeddings = self.projection(image_features)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.squeeze(0)


class FormulaEncoder(nn.Module):
    """MathBERT-based formula encoder for mathematical expressions"""
    
    def __init__(self, model_name: str = "MathBERT", 
                 embedding_dim: int = 512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, formula_latex: str, max_length: int = 256) -> torch.Tensor:
        """
        Args:
            formula_latex: LaTeX string of mathematical formula
            max_length: Maximum sequence length
        Returns:
            Normalized embedding vector [embedding_dim]
        """
        # Tokenize LaTeX
        inputs = self.tokenizer(
            formula_latex,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(next(self.encoder.parameters()).device)
        
        # Extract features
        outputs = self.encoder(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to unified space
        embeddings = self.projection(cls_output)
        
        # L2 normalization
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings.squeeze(0)


class MultiModalEncoder(nn.Module):
    """Unified multi-modal encoder"""
    
    def __init__(self, 
                 text_encoder_name: str = "scibert_scivocab_uncased",
                 image_encoder_name: str = "ViT-B-32",
                 formula_encoder_name: str = "MathBERT",
                 embedding_dim: int = 512):
        super().__init__()
        
        self.text_encoder = TextEncoder(text_encoder_name, embedding_dim)
        self.image_encoder = ImageEncoder(image_encoder_name, embedding_dim=embedding_dim)
        self.formula_encoder = FormulaEncoder(formula_encoder_name, embedding_dim)
        
        self.embedding_dim = embedding_dim
        
    def encode_text(self, text: str) -> torch.Tensor:
        return self.text_encoder(text)
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)
    
    def encode_formula(self, formula_latex: str) -> torch.Tensor:
        return self.formula_encoder(formula_latex)
    
    def forward(self, 
                text: Optional[str] = None,
                image: Optional[torch.Tensor] = None,
                formula: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities
        
        Returns:
            Dictionary with keys 'text', 'image', 'formula' containing embeddings
        """
        embeddings = {}
        
        if text is not None:
            embeddings['text'] = self.encode_text(text)
            
        if image is not None:
            embeddings['image'] = self.encode_image(image)
            
        if formula is not None:
            embeddings['formula'] = self.encode_formula(formula)
            
        return embeddings
