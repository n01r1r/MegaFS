"""
Modular training/validation/test infrastructure.
Easy to extend for custom experiments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm


class BaseTrainer:
    """
    Base trainer class providing train/val/test loop infrastructure.
    Inherit and override methods for custom experiments.
    """
    
    def __init__(self,
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: str = 'cuda',
                 log_dir: str = './logs'):
        """
        Args:
            model: MegaFS or wrapped model
            dataloaders: Dict with 'train', 'val', 'test' DataLoaders
            optimizer: PyTorch optimizer (optional, for training)
            device: Device to run on
            log_dir: Directory for saving logs/checkpoints
        """
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_epoch = 0
        self.global_step = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        Override this for custom training logic.
        
        Returns:
            Dict of metrics (e.g., {'loss': 0.5, 'accuracy': 0.9})
        """
        if hasattr(self.model, 'train'):
            self.model.train()
        elif hasattr(self.model, 'set_gradient_mode'):
            self.model.set_gradient_mode(True)
        
        metrics = {}
        total_loss = 0
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            source = batch['source'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            loss = self.compute_loss(source, target, batch)
            
            # Backward pass
            if self.optimizer:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Log metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            self.global_step += 1
        
        metrics['loss'] = total_loss / len(self.dataloaders['train'])
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        Override for custom validation logic.
        """
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, 'set_gradient_mode'):
            self.model.set_gradient_mode(False)
        
        metrics = {}
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc="Validation"):
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                loss = self.compute_loss(source, target, batch)
                total_loss += loss.item()
        
        metrics['val_loss'] = total_loss / len(self.dataloaders['val'])
        return metrics
    
    def test(self) -> Dict[str, float]:
        """
        Run testing.
        Override for custom test logic.
        """
        if hasattr(self.model, 'eval'):
            self.model.eval()
        elif hasattr(self.model, 'set_gradient_mode'):
            self.model.set_gradient_mode(False)
        
        metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc="Testing"):
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Compute metrics
                batch_metrics = self.compute_metrics(source, target, batch)
                
                # Aggregate metrics
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Average metrics
        for key in metrics:
            metrics[key] = sum(metrics[key]) / len(metrics[key])
        
        return metrics
    
    def compute_loss(self, source: torch.Tensor, target: torch.Tensor, 
                    batch: Dict) -> torch.Tensor:
        """
        Compute loss for a batch.
        MUST be overridden in subclasses for specific experiments.
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")
    
    def compute_metrics(self, source: torch.Tensor, target: torch.Tensor,
                       batch: Dict) -> Dict[str, float]:
        """
        Compute evaluation metrics for a batch.
        Override for custom metrics.
        """
        return {}
    
    def fit(self, num_epochs: int):
        """
        Full training loop with validation.
        
        Args:
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch} - Train: {train_metrics}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch} - Val: {val_metrics}")
            
            # Save checkpoint
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = self.log_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict() if hasattr(self.model, 'state_dict') else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
        
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        state = torch.load(checkpoint_path)
        
        self.current_epoch = state['epoch']
        self.global_step = state['global_step']
        
        if hasattr(self.model, 'load_state_dict') and state['model_state_dict']:
            self.model.load_state_dict(state['model_state_dict'])
        
        if self.optimizer and state['optimizer_state_dict']:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")


