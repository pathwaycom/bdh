import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments with continual learning options."""
    parser = argparse.ArgumentParser(description="Train BDH with optional continual learning")
    
    parser.add_argument('--model_name', type=str, default='BDH',
                       help='Name of the model')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='Hidden dimension size')
    parser.add_argument('--vocab_size', type=int, default=256,
                       help='Vocabulary size')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of model layers')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--continual_learning', action='store_true',
                       help='Enable continual learning mode')
    parser.add_argument('--num_tasks', type=int, default=1,
                       help='Number of tasks for continual learning')
    parser.add_argument('--cl_use_consolidation', action='store_true', default=True,
                       help='Use EWC consolidation')
    parser.add_argument('--cl_use_metaplasticity', action='store_true', default=True,
                       help='Use metaplasticity')
    parser.add_argument('--cl_use_adaptive_layers', action='store_true', default=True,
                       help='Use adaptive layers for continual learning')
    parser.add_argument('--cl_lambda_ewc', type=float, default=1000.0,
                       help='EWC regularization strength (lambda)')
    parser.add_argument('--cl_consolidation_rate', type=float, default=0.98,
                       help='Consolidation rate for metaplasticity')
    parser.add_argument('--cl_ewc_weight', type=float, default=0.5,
                       help='Weight of EWC loss in total loss')
    parser.add_argument('--cl_num_fisher_samples', type=int, default=200,
                       help='Number of samples for Fisher estimation')
    
    return parser.parse_args()


def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)


class BDHModel(nn.Module):
    """
    Baby Dragon Hatchling (BDH) Model with Continual Learning Support.
    
    Adapted from the original BDH architecture to support:
    - Adaptive layers for consolidation
    - Plasticity state tracking
    - Optional metaplasticity
    """
    
    def __init__(self, vocab_size, hidden_size, num_layers, 
                 use_adaptive_layers=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_adaptive_layers = use_adaptive_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_adaptive_layers:
                # Use adaptive linear layers for continual learning
                try:
                    from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
                    layer = nn.Sequential(
                        AdaptiveLinear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                except ImportError:
                    logger.warning("AdaptiveLinear not found, using standard Linear layers")
                    layer = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
            else:
                # Use standard layers
                layer = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
            self.layers.append(layer)
        
        # Output layer
        if use_adaptive_layers:
            try:
                from bdh.continual_learning.adaptive_synapses import AdaptiveLinear
                self.output_layer = AdaptiveLinear(hidden_size, vocab_size)
            except ImportError:
                self.output_layer = nn.Linear(hidden_size, vocab_size)
        else:
            self.output_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        """Forward pass through BDH model."""
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, hidden_size)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)  # (batch_size, seq_length, vocab_size)
        return x
    
    def forward_with_consolidation(self, x):
        """Forward pass with consolidation gating (if using adaptive layers)."""
        if not self.use_adaptive_layers:
            return self.forward(x)
        
        # Use consolidation gates
        x = self.embedding(x)
        for layer in self.layers:
            if hasattr(layer[0], 'forward_with_consolidation'):
                x = layer[0].forward_with_consolidation(x)
                x = layer[1](x)  # ReLU
            else:
                x = layer(x)
        
        if hasattr(self.output_layer, 'forward_with_consolidation'):
            x = self.output_layer.forward_with_consolidation(x)
        else:
            x = self.output_layer(x)
        
        return x


def create_dummy_tasks(num_tasks=2, batch_size=32, seq_length=64, vocab_size=256):
    """
    Create dummy tasks for testing continual learning.
    In practice, replace this with real task datasets.
    """
    tasks = []
    for task_id in range(num_tasks):
        # Create dummy data
        train_data = torch.randint(0, vocab_size, (100 * batch_size, seq_length))
        train_labels = torch.randint(0, vocab_size, (100 * batch_size, seq_length))
        val_data = torch.randint(0, vocab_size, (10 * batch_size, seq_length))
        val_labels = torch.randint(0, vocab_size, (10 * batch_size, seq_length))
        test_data = torch.randint(0, vocab_size, (10 * batch_size, seq_length))
        test_labels = torch.randint(0, vocab_size, (10 * batch_size, seq_length))
        
        # Create dataloaders
        train_loader = DataLoader(
            list(zip(train_data, train_labels)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            list(zip(val_data, val_labels)),
            batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            list(zip(test_data, test_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        tasks.append((train_loader, val_loader, test_loader))
    
    return tasks


def train_standard(model, train_loader, val_loader, loss_fn, optimizer, 
                  args, device):
    """Standard training without continual learning."""
    logger.info("Starting standard training (no continual learning)...")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = loss_fn(logits.view(-1, args.vocab_size), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits.view(-1, args.vocab_size), y.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        avg_train_loss = total_loss / len(train_loader)
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                       f"Val Loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break


def train_continual_learning(model, tasks, loss_fn, args, device):
    """Train with continual learning and consolidation."""
    logger.info(f"Starting continual learning training with {len(tasks)} tasks...")
    
    # Import continual learning modules
    from bdh.continual_learning.trainer import (
        train_continual_learning_task,
        continual_learning_training_loop
    )
    from bdh.continual_learning.metrics import ContinualLearningMetrics
    
    # Train with continual learning
    accuracies = continual_learning_training_loop(
        model=model,
        tasks=tasks,
        num_epochs_per_task=args.epochs,
        lr=args.learning_rate,
        use_consolidation=args.cl_use_consolidation,
        use_metaplasticity=args.cl_use_metaplasticity
    )
    
    # Evaluate and print metrics
    metrics = ContinualLearningMetrics(accuracies)
    metrics.print_summary()
    
    return metrics


def main():
    """Main training function."""
    args = parse_arguments()
    setup_seed(args.seed)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create model
    logger.info("Creating BDH model...")
    model = BDHModel(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_adaptive_layers=args.continual_learning and args.cl_use_adaptive_layers
    )
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    logger.info(f"  - Hidden size: {args.hidden_size}")
    logger.info(f"  - Num layers: {args.num_layers}")
    logger.info(f"  - Vocab size: {args.vocab_size}")
    logger.info(f"  - Adaptive layers: {args.cl_use_adaptive_layers}")
    
    # ===== MAIN TRAINING DECISION =====
    if args.continual_learning:
        logger.info("=" * 60)
        logger.info("CONTINUAL LEARNING MODE")
        logger.info("=" * 60)
        
        # Create tasks
        logger.info(f"Creating {args.num_tasks} tasks...")
        tasks = create_dummy_tasks(
            num_tasks=args.num_tasks,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size
        )
        
        # Log continual learning settings
        logger.info(f"CL Configuration:")
        logger.info(f"  - Use consolidation: {args.cl_use_consolidation}")
        logger.info(f"  - Use metaplasticity: {args.cl_use_metaplasticity}")
        logger.info(f"  - Lambda EWC: {args.cl_lambda_ewc}")
        logger.info(f"  - Consolidation rate: {args.cl_consolidation_rate}")
        logger.info(f"  - EWC weight: {args.cl_ewc_weight}")
        
        # Train with continual learning
        loss_fn = nn.CrossEntropyLoss()
        metrics = train_continual_learning(
            model, tasks, loss_fn, args, device
        )
        
        # Save results
        results_file = os.path.join(args.checkpoint_dir, 'cl_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'average_accuracy': metrics.average_accuracy(),
                'backward_transfer': metrics.backward_transfer(),
                'num_tasks': args.num_tasks,
            }, f, indent=4)
        logger.info(f"Results saved to {results_file}")
    
    else:
        logger.info("=" * 60)
        logger.info("STANDARD TRAINING MODE (No Continual Learning)")
        logger.info("=" * 60)
        
        # Create standard training data (dummy for demo)
        logger.info("Creating training data...")
        train_data = torch.randint(0, args.vocab_size, (1000, 64))
        train_labels = torch.randint(0, args.vocab_size, (1000, 64))
        val_data = torch.randint(0, args.vocab_size, (100, 64))
        val_labels = torch.randint(0, args.vocab_size, (100, 64))
        
        train_loader = DataLoader(
            list(zip(train_data, train_labels)),
            batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            list(zip(val_data, val_labels)),
            batch_size=args.batch_size, shuffle=False
        )
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), 
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        
        # Train
        train_standard(model, train_loader, val_loader, loss_fn, 
                      optimizer, args, device)
    
    # Save final model
    model_path = os.path.join(args.checkpoint_dir, f'{args.model_name}_final.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
