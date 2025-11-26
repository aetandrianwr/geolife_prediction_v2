"""
Train Ultra-Advanced Model with Hierarchical Loss

Multi-task learning:
- Location prediction (primary)
- Cluster prediction (auxiliary)
- Combined with smart weighting
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.ultra_model import create_ultra_model
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict
from src.utils.logger import setup_logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HierarchicalLoss(nn.Module):
    """Hierarchical loss: location + cluster prediction."""
    
    def __init__(self, location_weight=0.7, cluster_weight=0.3):
        super().__init__()
        self.location_weight = location_weight
        self.cluster_weight = cluster_weight
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'logits', 'cluster_logits'
            targets: ground truth location indices
        """
        # Location loss
        location_loss = self.ce_loss(outputs['logits'], targets)
        
        # Cluster loss (soft targets from cluster assignments)
        cluster_targets = outputs['cluster_targets']
        cluster_probs = F.softmax(outputs['cluster_logits'], dim=-1)
        cluster_loss = -torch.sum(cluster_targets * torch.log(cluster_probs + 1e-10), dim=-1).mean()
        
        # Combined loss
        total_loss = self.location_weight * location_loss + self.cluster_weight * cluster_loss
        
        return total_loss, {
            'location_loss': location_loss.item(),
            'cluster_loss': cluster_loss.item(),
            'total_loss': total_loss.item()
        }


class UltraTrainer:
    """Trainer for ultra-advanced model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device='cuda',
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=200,
        patience=40,
        warmup_epochs=15,
        checkpoint_dir='checkpoints/ultra_model',
        logger=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or setup_logger('ultra', 'logs/ultra_training.log')
        
        # Loss function
        self.criterion = HierarchicalLoss()
        
        # Optimizer with layer-wise learning rates
        param_groups = [
            {'params': self.model.location_embed.parameters(), 'lr': learning_rate * 0.5},
            {'params': self.model.user_embed.parameters(), 'lr': learning_rate * 0.5},
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'embed' not in n], 'lr': learning_rate}
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Cosine annealing with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
        )
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def warmup_lr(self, epoch):
        """Linear warmup for learning rate."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.max_epochs}')
        for batch in pbar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward
            outputs = self.model(batch)
            loss, loss_dict = self.criterion(outputs, batch['target'])
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = outputs['logits'].argmax(dim=1)
            correct = (preds == batch['target']).sum().item()
            total_correct += correct
            total_samples += batch['target'].size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*correct/batch['target'].size(0):.1f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        all_logits = []
        all_targets = []
        
        for batch in tqdm(loader, desc='Evaluating'):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            outputs = self.model(batch)
            all_logits.append(outputs['logits'].cpu())
            all_targets.append(batch['target'].cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics_array, _, _ = calculate_correct_total_prediction(all_logits, all_targets)
        perf = get_performance_dict({
            'correct@1': metrics_array[0],
            'correct@3': metrics_array[1],
            'correct@5': metrics_array[2],
            'correct@10': metrics_array[3],
            'rr': metrics_array[4],
            'ndcg': metrics_array[5],
            'f1': 0.0,
            'total': metrics_array[6]
        })
        
        return perf
    
    def train(self):
        """Full training loop."""
        self.logger.info("="*80)
        self.logger.info("ULTRA-ADVANCED MODEL TRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Parameters: {self.model.count_parameters():,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("="*80)
        
        for epoch in range(1, self.max_epochs + 1):
            # Warmup
            if epoch <= self.warmup_epochs:
                self.warmup_lr(epoch)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_perf = self.evaluate(self.val_loader)
            
            # Scheduler step (after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            
            # Log
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Acc@1: {val_perf['acc@1']:.2f}% | "
                f"Val Acc@5: {val_perf['acc@5']:.2f}% | "
                f"LR: {lr:.2e}"
            )
            
            # Save best model
            if val_perf['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_perf['acc@1']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_perf['acc@1']
                }, self.checkpoint_dir / 'best_model.pt')
                
                self.logger.info(f"✓ New best model! Val Acc@1: {val_perf['acc@1']:.2f}%")
            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"  No improvement for {self.epochs_without_improvement} epochs (Best: {self.best_val_acc:.2f}%)")
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model and evaluate on test
        self.logger.info("\n" + "="*80)
        self.logger.info("Loading best model for final evaluation...")
        checkpoint = torch.load(self.checkpoint_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_perf = self.evaluate(self.test_loader)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL TEST RESULTS:")
        self.logger.info(f"  Test Acc@1:  {test_perf['acc@1']:.2f}%")
        self.logger.info(f"  Test Acc@5:  {test_perf['acc@5']:.2f}%")
        self.logger.info(f"  Test Acc@10: {test_perf['acc@10']:.2f}%")
        self.logger.info(f"  Test MRR:    {test_perf['mrr']:.2f}%")
        self.logger.info(f"  Best Val Acc@1: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        self.logger.info("="*80)
        
        # Goal assessment
        self.logger.info("\n" + "="*80)
        self.logger.info("GOAL ASSESSMENT")
        self.logger.info("="*80)
        if test_perf['acc@1'] >= 50.0:
            self.logger.info(f"🎉 EXCELLENT! {test_perf['acc@1']:.2f}% (Target: >50%)")
        elif test_perf['acc@1'] >= 40.0:
            self.logger.info(f"✓ SUCCESS! {test_perf['acc@1']:.2f}% (Minimum: 40%)")
            self.logger.info(f"  Gap to 50%: {50.0 - test_perf['acc@1']:.2f}%")
        else:
            self.logger.info(f"⚠ Below target: {test_perf['acc@1']:.2f}%")
            self.logger.info(f"  Need: {40.0 - test_perf['acc@1']:.2f}% more for minimum")
        self.logger.info("="*80)
        
        return test_perf


def main():
    set_seed(2024)  # Different seed for diversity
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = '/content/geolife_prediction_v2/data/geolife'
    
    logger = setup_logger('ultra', 'logs/ultra_model.log')
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Create ultra model
    logger.info("Creating ultra-advanced model...")
    model = create_ultra_model(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users']
    )
    
    # Train
    trainer = UltraTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_epochs=200,
        patience=40,
        warmup_epochs=15,
        logger=logger
    )
    
    test_perf = trainer.train()
    
    # Save results
    with open('results/ultra_model_results.txt', 'w') as f:
        f.write("ULTRA-ADVANCED MODEL RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write("Architecture:\n")
        f.write("  - Hybrid LSTM + Transformer\n")
        f.write("  - Fourier + Rotational encodings\n")
        f.write("  - Multi-scale temporal attention\n")
        f.write("  - Learnable location clustering\n")
        f.write("  - Time-interval encoding\n\n")
        f.write(f"Parameters: {model.count_parameters():,}\n\n")
        f.write(f"Test Acc@1:  {test_perf['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5:  {test_perf['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_perf['acc@10']:.2f}%\n")
        f.write(f"Test MRR:    {test_perf['mrr']:.2f}%\n")
    
    logger.info("\n✓ Results saved to results/ultra_model_results.txt")
    
    return test_perf


if __name__ == '__main__':
    main()
