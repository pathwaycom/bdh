import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Callable
import tqdm

from .consolidation import ElasticWeightConsolidation, MetaplasticityConsolidation
from .importance import HybridImportanceEstimator
from .hebbian import HebbianPlasticity


def train_epoch(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: optim.Optimizer,
               loss_fn: nn.Module,
               ewc_loss_fn: ElasticWeightConsolidation,
               importance_est: HybridImportanceEstimator,
               ewc_weight: float = 1.0,
               use_consolidation: bool = False) -> float:
    """
    Single training epoch with consolidation.
    
    Args:
        model: Neural network
        data_loader: Training data
        optimizer: Optimizer
        loss_fn: Loss function
        ewc_loss_fn: EWC loss
        importance_est: Importance estimator
        ewc_weight: Weight for EWC loss
        use_consolidation: Whether to use consolidation gates
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm.tqdm(data_loader, desc="Training")
    
    for x, y in pbar:
        # Forward pass
        if use_consolidation:
            # Use consolidation gates
            logits = model(x)  # Model should support consolidation internally
        else:
            logits = model(x)
        
        task_loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # EWC regularization
        ewc_loss = ewc_loss_fn(model)
        
        total_loss_batch = task_loss + ewc_weight * ewc_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss_batch.backward()
        
        # Update importance estimate (online)
        importance_est.get_online_importance(task_loss)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / (num_batches + 1)})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model: nn.Module,
            data_loader: torch.utils.data.DataLoader) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)  # logits shape: (batch_size, seq_len, vocab_size)
            pred = logits.argmax(dim=-1)  # pred shape: (batch_size, seq_len)
            
            # Flatten predictions and labels for element-wise comparison
            pred_flat = pred.view(-1)
            y_flat = y.view(-1)
            
            correct += (pred_flat == y_flat).sum().item()
            total += y_flat.size(0)
    
    return correct / total if total > 0 else 0.0


def train_continual_learning_task(model: nn.Module,
                                 train_loader: torch.utils.data.DataLoader,
                                 val_loader: torch.utils.data.DataLoader,
                                 loss_fn: nn.Module,
                                 ewc_loss_fn: ElasticWeightConsolidation,
                                 importance_est: HybridImportanceEstimator,
                                 num_epochs: int = 40,
                                 lr: float = 0.001,
                                 use_consolidation: bool = True) -> float:
    """
    Train on single task with early stopping.
    
    Returns:
        Best validation accuracy
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print(f"\nTraining with {len(train_loader)} batches...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, ewc_loss_fn,
            importance_est, ewc_weight=0.5 if not ewc_loss_fn.is_first_task else 0.0,
            use_consolidation=use_consolidation
        )
        
        val_acc = evaluate(model, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs}: Loss={train_loss:.4f}, "
                  f"Val_Acc={val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return best_val_acc


def continual_learning_training_loop(model: nn.Module,
                                    tasks: List[Tuple],
                                    num_epochs_per_task: int = 40,
                                    lr: float = 0.001,
                                    use_consolidation: bool = True,
                                    use_metaplasticity: bool = True) -> List[List[float]]:
    """
    Main continual learning loop.
    
    Args:
        model: Neural network
        tasks: List of (train_loader, val_loader, test_loader) tuples
        num_epochs_per_task: Training epochs per task
        lr: Learning rate
        use_consolidation: Use EWC consolidation
        use_metaplasticity: Use metaplasticity
    
    Returns:
        Accuracy matrix: accuracies[task_id][eval_task_id]
    """
    loss_fn = nn.CrossEntropyLoss()
    ewc_loss_fn = ElasticWeightConsolidation(lambda_ewc=1000.0)
    importance_est = HybridImportanceEstimator(model)
    metaplasticity = MetaplasticityConsolidation(consolidation_rate=0.98)
    
    accuracies = []
    
    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_id}")
        print(f"{'='*60}")
        
        # Train on task
        train_continual_learning_task(
            model, train_loader, val_loader, loss_fn, ewc_loss_fn,
            importance_est, num_epochs=num_epochs_per_task, lr=lr,
            use_consolidation=use_consolidation
        )
        
        # Consolidate task
        if use_consolidation:
            print(f"\nConsolidating Task {task_id}...")
            
            # Estimate Fisher information
            fisher_dict = importance_est.get_task_importance(
                train_loader, loss_fn
            )
            
            # Save weights and Fisher
            ewc_loss_fn.consolidate_task(model, fisher_dict)
            
            # Apply metaplasticity
            if use_metaplasticity:
                for cons_epoch in range(10):
                    metaplasticity.apply_consolidation(model, strength=0.9)
        
        # Test on all previous tasks
        task_accs = []
        for eval_task_id in range(task_id + 1):
            _, _, test_loader_eval = tasks[eval_task_id]
            acc = evaluate(model, test_loader_eval)
            task_accs.append(acc)
            print(f"  Task {eval_task_id}: Accuracy={acc:.4f}")
        
        accuracies.append(task_accs)
    
    return accuracies
