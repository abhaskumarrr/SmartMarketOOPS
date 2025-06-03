"""
Memory-Efficient Transformer Implementation for M2 MacBook Air 8GB
Task #24: Subtask 24.4 - Performance Optimization and Memory Management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gc
from typing import Optional, Dict, Any, Tuple
from .transformer_model import EnhancedTransformerModel
import logging

logger = logging.getLogger(__name__)

# Memory-efficient configuration for M2 MacBook Air 8GB
MEMORY_CONFIG = {
    'max_batch_size': 16,  # Reduced for 8GB RAM
    'gradient_accumulation_steps': 4,  # Simulate larger batches
    'mixed_precision': True,
    'gradient_checkpointing': True,
    'attention_dropout': 0.15,  # Slightly higher for regularization
    'layer_dropout': 0.1,
    'max_sequence_length': 100,  # Limit sequence length
    'model_parallel': False,  # Single GPU/CPU optimization
    'memory_cleanup_frequency': 10  # Clean memory every N batches
}


class MemoryEfficientTransformer(EnhancedTransformerModel):
    """
    Memory-optimized Transformer for resource-constrained environments
    Implements gradient checkpointing, mixed precision, and memory management
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        forecast_horizon: int,
        d_model: int = 128,  # Reduced from 256 for memory efficiency
        nhead: int = 8,
        num_layers: int = 4,  # Reduced from 6 for memory efficiency
        dropout: float = 0.15,
        device: Optional[str] = None,
        use_gradient_checkpointing: bool = True,
        use_mixed_precision: bool = True,
        **kwargs
    ):
        # Adjust parameters for memory efficiency
        memory_optimized_kwargs = {
            'd_model': min(d_model, 128),  # Cap at 128 for memory
            'nhead': min(nhead, 8),
            'num_layers': min(num_layers, 4),
            'dropout': dropout,
            'use_financial_attention': kwargs.get('use_financial_attention', True)
        }
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=min(seq_len, MEMORY_CONFIG['max_sequence_length']),
            forecast_horizon=forecast_horizon,
            device=device,
            **memory_optimized_kwargs
        )
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = MEMORY_CONFIG['gradient_accumulation_steps']
        self.memory_cleanup_frequency = MEMORY_CONFIG['memory_cleanup_frequency']
        
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Mixed precision scaler
        if use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        logger.info(f"Memory-efficient Transformer initialized: "
                   f"d_model={self.d_model}, layers={self.num_layers}, "
                   f"gradient_checkpointing={use_gradient_checkpointing}, "
                   f"mixed_precision={use_mixed_precision}")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self, 'transformer_layers'):
            for layer in self.transformer_layers:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient forward pass"""
        # Ensure input is within memory limits
        if x.size(1) > MEMORY_CONFIG['max_sequence_length']:
            x = x[:, -MEMORY_CONFIG['max_sequence_length']:, :]
            logger.warning(f"Truncated sequence to {MEMORY_CONFIG['max_sequence_length']} for memory efficiency")
        
        # Use gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, mask)
        else:
            return super().forward(x, mask)
    
    def _forward_with_checkpointing(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with gradient checkpointing"""
        # Input projection
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Apply transformer layers with checkpointing
        if self.use_financial_attention:
            for layer in self.transformer_layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, mask, use_reentrant=False)
        else:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, mask, use_reentrant=False)
        
        # Layer normalization and output projection
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        output = self.output_projection(x)
        
        return output
    
    def fit_model_memory_efficient(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,  # Reduced default epochs
        lr: float = 0.0005,  # Slightly lower learning rate
        early_stopping_patience: int = 8,
        checkpoint_dir: Optional[str] = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Memory-efficient training with gradient accumulation and mixed precision
        """
        # Setup loss function
        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device) if class_weights is not None else None
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )
        
        # Setup optimizer with memory-efficient settings
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training tracking
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting memory-efficient training for {num_epochs} epochs")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            epoch_train_losses = []
            optimizer.zero_grad()
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X = X.to(self.device).float()
                y = y.to(self.device).float() if self.output_dim == 1 else y.to(self.device).long()
                
                # Mixed precision forward pass
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self(X)
                        if self.output_dim == 1:
                            outputs = outputs.squeeze(-1)
                            loss = criterion(outputs, y)
                        else:
                            loss = criterion(outputs, y)
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    
                    # Update weights every gradient_accumulation_steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad()
                
                else:
                    # Standard precision training
                    outputs = self(X)
                    if self.output_dim == 1:
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, y)
                    else:
                        loss = criterion(outputs, y)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    # Update weights every gradient_accumulation_steps
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_train_losses.append(loss.item() * self.gradient_accumulation_steps)
                
                # Memory cleanup
                if batch_idx % self.memory_cleanup_frequency == 0:
                    self._cleanup_memory()
            
            # Validation phase
            val_loss = self._evaluate_loss_memory_efficient(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Track losses
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.state_dict().copy()
                
                # Save best model
                if checkpoint_dir:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(self.state_dict(), 
                             os.path.join(checkpoint_dir, "best_memory_efficient_transformer.pt"))
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}: New best validation loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch+1}/{num_epochs}: Val loss: {val_loss:.6f}, "
                           f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Memory cleanup after each epoch
            self._cleanup_memory()
        
        # Load best model
        if best_state:
            self.load_state_dict(best_state)
            logger.info("Loaded best model state")
        
        return {
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epochs_trained": epoch + 1,
            "memory_config": MEMORY_CONFIG
        }
    
    def _evaluate_loss_memory_efficient(self, loader, criterion):
        """Memory-efficient evaluation"""
        self.eval()
        losses = []
        
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device).float()
                y = y.to(self.device).float() if self.output_dim == 1 else y.to(self.device).long()
                
                # Process in smaller chunks if needed
                if X.size(0) > MEMORY_CONFIG['max_batch_size']:
                    chunk_losses = []
                    for i in range(0, X.size(0), MEMORY_CONFIG['max_batch_size']):
                        chunk_X = X[i:i+MEMORY_CONFIG['max_batch_size']]
                        chunk_y = y[i:i+MEMORY_CONFIG['max_batch_size']]
                        
                        outputs = self(chunk_X)
                        if self.output_dim == 1:
                            outputs = outputs.squeeze(-1)
                            loss = criterion(outputs, chunk_y)
                        else:
                            loss = criterion(outputs, chunk_y)
                        chunk_losses.append(loss.item())
                    
                    losses.append(np.mean(chunk_losses))
                else:
                    outputs = self(X)
                    if self.output_dim == 1:
                        outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, y)
                    else:
                        loss = criterion(outputs, y)
                    losses.append(loss.item())
        
        return np.mean(losses)
    
    def _cleanup_memory(self):
        """Clean up GPU/CPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        import psutil
        process = psutil.Process()
        stats['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        
        return stats
