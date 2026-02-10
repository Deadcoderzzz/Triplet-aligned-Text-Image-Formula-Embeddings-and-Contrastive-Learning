import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

from tif_contrast.models.tif_model import TIFContrast
from tif_contrast.losses.contrastive import TIFContrastLoss
from tif_contrast.data.dataset import AcademicDocumentDataset, collate_fn
from tif_contrast.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train TIF-Contrast model')
    parser.add_argument('--config', type=str, default='configs/tif_contrast.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Path to validation data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lambda1', type=float, default=0.6,
                       help='Weight for pairwise loss')
    parser.add_argument('--lambda2', type=float, default=0.4,
                       help='Weight for triplet loss')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    pairwise_loss_sum = 0
    triplet_loss_sum = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (doc1_batch, doc2_batch, labels) in enumerate(pbar):
        # Move to device
        doc1_images = doc1_batch['image'].to(device)
        doc2_images = doc2_batch['image'].to(device)
        
        # Forward pass - process batch
        batch_size = len(doc1_batch['text'])
        
        modal_embeddings_list = {'text': [], 'image': [], 'formula': []}
        fused_embeddings_list = []
        
        for i in range(batch_size):
            # Encode doc1
            doc1_modal, doc1_fused = model.encode_document(
                text=doc1_batch['text'][i],
                image=doc1_images[i],
                formula=doc1_batch['formula'][i]
            )
            
            # Collect embeddings
            for key in modal_embeddings_list:
                if key in doc1_modal:
                    modal_embeddings_list[key].append(doc1_modal[key])
            
            fused_embeddings_list.append(doc1_fused)
        
        # Stack embeddings
        modal_embeddings = {
            key: torch.stack(embs) for key, embs in modal_embeddings_list.items() if embs
        }
        fused_embeddings = torch.stack(fused_embeddings_list)
        
        # Compute loss
        losses = criterion(modal_embeddings, fused_embeddings)
        loss = losses['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pairwise_loss_sum += losses['pairwise'].item()
        triplet_loss_sum += losses['triplet'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'pair': f"{losses['pairwise'].item():.4f}",
            'triplet': f"{losses['triplet'].item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_pairwise = pairwise_loss_sum / len(dataloader)
    avg_triplet = triplet_loss_sum / len(dataloader)
    
    return {
        'loss': avg_loss,
        'pairwise_loss': avg_pairwise,
        'triplet_loss': avg_triplet
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for doc1_batch, doc2_batch, labels in tqdm(dataloader, desc='Validating'):
        doc1_images = doc1_batch['image'].to(device)
        doc2_images = doc2_batch['image'].to(device)
        
        batch_size = len(doc1_batch['text'])
        similarities = []
        
        for i in range(batch_size):
            # Compute similarity
            doc1 = {
                'text': doc1_batch['text'][i],
                'image': doc1_images[i],
                'formula': doc1_batch['formula'][i]
            }
            doc2 = {
                'text': doc2_batch['text'][i],
                'image': doc2_images[i],
                'formula': doc2_batch['formula'][i]
            }
            
            similarity = model.compute_similarity(doc1, doc2)
            similarities.append(similarity)
        
        similarities = torch.tensor(similarities)
        predictions = (similarities > 0.5).long()
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions)
    
    return metrics


def main():
    args = parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project='tif-contrast', config=vars(args))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = TIFContrast(
        embedding_dim=config.get('embedding_dim', 512),
        use_adaptive_fusion=config.get('use_adaptive_fusion', True),
        modality_dropout=config.get('modality_dropout', 0.1)
    ).to(device)
    
    # Multi-GPU
    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    # Create datasets
    train_dataset = AcademicDocumentDataset(
        data_dir=args.data_dir,
        pairs_file=os.path.join(args.data_dir, 'pairs.csv')
    )
    
    val_dataset = AcademicDocumentDataset(
        data_dir=args.val_dir,
        pairs_file=os.path.join(args.val_dir, 'pairs.csv')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Loss function
    criterion = TIFContrastLoss(
        lambda_pair=args.lambda1,
        lambda_triplet=args.lambda2,
        temperature=0.07
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    warmup_epochs = int(0.1 * args.epochs)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    patience = 20
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall']
            })
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            save_path = os.path.join(args.output_dir, 'tif-contrast-best.pth')
            model.save_checkpoint(
                save_path,
                optimizer=optimizer,
                epoch=epoch,
                best_f1=best_f1
            )
            print(f"Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            model.save_checkpoint(save_path, optimizer=optimizer, epoch=epoch)
    
    print("Training completed!")
    print(f"Best F1 score: {best_f1:.4f}")


if __name__ == '__main__':
    main()
