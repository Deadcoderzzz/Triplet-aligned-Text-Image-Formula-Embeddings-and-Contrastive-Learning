import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletAlignmentLoss(nn.Module):

    def __init__(self, margin: float = 0.3):
        super(TripletAlignmentLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(anchor, positive, negative)

class CrossModalTripletLoss(nn.Module):
    """
    Computes all-pairs triplet losses across three modalities.
    """
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.base_loss = TripletAlignmentLoss(margin)

    def forward(self, text_emb, img_emb, formula_emb, neg_img, neg_formula):
        # Text as anchor, aligning Image and Formula
        l_ti = self.base_loss(text_emb, img_emb, neg_img)
        l_tf = self.base_loss(text_emb, formula_emb, neg_formula)
        return (l_ti + l_tf) / 2