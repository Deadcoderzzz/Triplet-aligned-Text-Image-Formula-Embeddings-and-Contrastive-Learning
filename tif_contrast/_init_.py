from .models.tif_model import TIFContrast
from .inference import TIFInference, batch_inference, compare_documents

__all__ = [
    'TIFContrast',
    'TIFInference',
    'batch_inference',
    'compare_documents'
]

from .tif_model import TIFContrast
from .encoders import MultiModalEncoder, TextEncoder, ImageEncoder, FormulaEncoder
from .fusion import AdaptiveAttentionFusion, InterpretableAttention

__all__ = [
    'TIFContrast',
    'MultiModalEncoder',
    'TextEncoder',
    'ImageEncoder', 
    'FormulaEncoder',
    'AdaptiveAttentionFusion',
    'InterpretableAttention'
]

from .contrastive import (
    InfoNCELoss,
    PairwiseAlignmentLoss,
    TripletAlignmentLoss,
    TIFContrastLoss
)

__all__ = [
    'InfoNCELoss',
    'PairwiseAlignmentLoss',
    'TripletAlignmentLoss',
    'TIFContrastLoss'
]

from .dataset import AcademicDocumentDataset, TripletDataset, collate_fn

__all__ = [
    'AcademicDocumentDataset',
    'TripletDataset',
    'collate_fn'
]

from .metrics import (
    compute_metrics,
    compute_retrieval_metrics,
    plot_confusion_matrix,
    plot_recall_curve,
    print_classification_report,
    analyze_errors
)

__all__ = [
    'compute_metrics',
    'compute_retrieval_metrics',
    'plot_confusion_matrix',
    'plot_recall_curve',
    'print_classification_report',
    'analyze_errors'
]
