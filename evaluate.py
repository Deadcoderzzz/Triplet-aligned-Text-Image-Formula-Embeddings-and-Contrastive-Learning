import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from tif_contrast.models.tif_model import TIFContrast
from tif_contrast.data.dataset import AcademicDocumentDataset, collate_fn
from tif_contrast.utils.metrics import compute_metrics, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TIF-Contrast model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_file', type=str, default='results/eval_results.json')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for classification')
    parser.add_argument('--ablation', type=str, default=None,
                       choices=['without_text', 'without_image', 'without_formula',
                               'without_triplet', 'without_adaptive'],
                       help='Ablation study variant')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.5, ablation=None):
    model.eval()
    
    all_similarities = []
    all_predictions = []
    all_labels = []
    attention_weights_list = []
    
    for doc1_batch, doc2_batch, labels in tqdm(dataloader, desc='Evaluating'):
        doc1_images = doc1_batch['image'].to(device)
        doc2_images = doc2_batch['image'].to(device)
        
        batch_size = len(doc1_batch['text'])
        
        for i in range(batch_size):
            # Prepare documents based on ablation
            doc1 = {
                'text': doc1_batch['text'][i] if ablation != 'without_text' else None,
                'image': doc1_images[i] if ablation != 'without_image' else None,
                'formula': doc1_batch['formula'][i] if ablation != 'without_formula' else None
            }
            doc2 = {
                'text': doc2_batch['text'][i] if ablation != 'without_text' else None,
                'image': doc2_images[i] if ablation != 'without_image' else None,
                'formula': doc2_batch['formula'][i] if ablation != 'without_formula' else None
            }
            
            # Compute similarity
            similarity = model.compute_similarity(doc1, doc2)
            prediction = 1 if similarity >= threshold else 0
            
            all_similarities.append(similarity)
            all_predictions.append(prediction)
            all_labels.append(labels[i].item())
            
            # Get attention weights for interpretability
            if ablation is None:  # Only for full model
                weights = model.get_attention_weights(
                    text=doc1['text'],
                    image=doc1['image'],
                    formula=doc1['formula']
                )
                attention_weights_list.append(weights)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions)
    
    # Add similarity scores
    metrics['similarities'] = all_similarities
    metrics['predictions'] = all_predictions
    metrics['labels'] = all_labels
    
    if attention_weights_list:
        # Average attention weights
        avg_weights = {}
        for key in attention_weights_list[0].keys():
            avg_weights[key] = np.mean([w[key] for w in attention_weights_list])
        metrics['avg_attention_weights'] = avg_weights
    
    return metrics


def run_ablation_studies(model, dataloader, device):
    """Run all ablation experiments"""
    
    ablation_variants = [
        None,  # Full model
        'without_text',
        'without_image',
        'without_formula',
        'without_adaptive'
    ]
    
    results = {}
    
    for variant in ablation_variants:
        print(f"\n{'='*50}")
        print(f"Evaluating: {variant if variant else 'Full Model'}")
        print(f"{'='*50}")
        
        # Modify model for ablation
        if variant == 'without_adaptive':
            original_fusion = model.use_adaptive_fusion
            model.use_adaptive_fusion = False
        
        metrics = evaluate_model(model, dataloader, device, ablation=variant)
        
        # Restore model
        if variant == 'without_adaptive':
            model.use_adaptive_fusion = original_fusion
        
        variant_name = variant if variant else 'full_model'
        results[variant_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = TIFContrast.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    test_dataset = AcademicDocumentDataset(
        data_dir=args.test_dir,
        pairs_file=os.path.join(args.test_dir, 'pairs.csv')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run evaluation
    if args.ablation:
        # Single ablation experiment
        print(f"\nRunning ablation: {args.ablation}")
        metrics = evaluate_model(
            model, test_loader, device,
            threshold=args.threshold,
            ablation=args.ablation
        )
        
        results = {
            'ablation': args.ablation,
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
        }
    else:
        # Full evaluation
        print("\nRunning full evaluation...")
        metrics = evaluate_model(
            model, test_loader, device,
            threshold=args.threshold
        )
        
        results = {
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            },
            'threshold': args.threshold
        }
        
        if 'avg_attention_weights' in metrics:
            results['avg_attention_weights'] = metrics['avg_attention_weights']
            print("\nAverage Attention Weights:")
            for modality, weight in metrics['avg_attention_weights'].items():
                print(f"  {modality}: {weight:.4f}")
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall:    {results['metrics']['recall']:.4f}")
    print(f"F1-score:  {results['metrics']['f1']:.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")
    
    # Save predictions if requested
    if args.save_predictions and not args.ablation:
        predictions_file = args.output_file.replace('.json', '_predictions.json')
        predictions_data = {
            'similarities': metrics['similarities'],
            'predictions': metrics['predictions'],
            'labels': metrics['labels']
        }
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"Predictions saved to {predictions_file}")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            metrics['labels'],
            metrics['predictions'],
            save_path=args.output_file.replace('.json', '_confusion_matrix.png')
        )


if __name__ == '__main__':
    main()
