"""
Example showing how to extend BaseTrainer for custom experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .trainer import BaseTrainer


class ExampleExperiment(BaseTrainer):
    """
    Example experiment extending base trainer.
    Shows how to implement custom training logic.
    """
    
    def __init__(self, model, dataloaders, **kwargs):
        super().__init__(model, dataloaders, **kwargs)
        
        # Enable gradients on MegaFS
        if hasattr(model, 'set_gradient_mode'):
            model.set_gradient_mode(True)
    
    def compute_loss(self, source, target, batch):
        """
        Example loss computation.
        Replace with your custom loss.
        """
        # Forward pass through model
        if hasattr(self.model, 'forward'):
            output = self.model.forward(source, target)
        else:
            # Fallback to swap method
            output = self.model.swap(source, target, return_tensor=True)
        
        # Simple L2 loss as example
        # Note: This is just for demonstration
        # In real experiments, you would use appropriate loss functions
        loss = F.mse_loss(output, target)
        
        return loss
    
    def compute_metrics(self, source, target, batch):
        """
        Example metrics computation.
        """
        if hasattr(self.model, 'forward'):
            output = self.model.forward(source, target)
        else:
            output = self.model.swap(source, target, return_tensor=True)
        
        # Example: compute PSNR
        mse = F.mse_loss(output, target)
        psnr = 10 * torch.log10(1.0 / mse)
        
        return {'psnr': psnr.item()}


# Usage example:
def run_experiment():
    """
    Example of how to use the trainer.
    """
    from config import Config
    from data.face_swap_dataset import create_dataloaders
    from models.megafs import MegaFS
    
    # Load config
    config = Config.from_yaml('configs/experiment.yaml')
    
    # Create model
    model = MegaFS(config=config, enable_grads=True)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_map_path='./data_map.json',
        dataset_root='./dataset/CelebAMask-HQ',
        batch_size=4
    )
    
    # Create optimizer (optional)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = ExampleExperiment(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device='cuda'
    )
    
    # Run experiment
    trainer.fit(num_epochs=10)
    
    # Test
    test_metrics = trainer.test()
    print(f"Test metrics: {test_metrics}")
    
    return trainer


