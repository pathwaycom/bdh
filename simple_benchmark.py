import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import numpy as np
from datetime import datetime

from benchmarks_complete import (
    PermutedMNISTGenerator,
    SplitCIFARGenerator,
    RotatedMNISTGenerator,
    ImprovedSequenceGenerator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_benchmark(benchmark_name: str, num_tasks: int):
    if benchmark_name == 'permuted_mnist':
        gen = PermutedMNISTGenerator(num_tasks=num_tasks, samples_per_task=1000)
        input_size = 784
        output_size = 10
    elif benchmark_name == 'split_cifar':
        gen = SplitCIFARGenerator(num_tasks=num_tasks, samples_per_class=500)
        input_size = 3072
        output_size = 2
    elif benchmark_name == 'rotated_mnist':
        gen = RotatedMNISTGenerator(num_tasks=num_tasks, samples_per_task=1000)
        input_size = 784
        output_size = 10
    elif benchmark_name == 'sequence':
        gen = ImprovedSequenceGenerator(num_tasks=num_tasks, samples_per_task=2000)
        input_size = 64
        output_size = 10
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return gen.get_all_tasks(batch_size=32), input_size, output_size

class SimpleModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

def train_and_evaluate(args):
    device = torch.device(args.device)
    logger.info("=" * 70)
    logger.info(f"BENCHMARK: {args.benchmark.upper()}")
    logger.info(f"TASKS: {args.num_tasks}, EPOCHS: {args.epochs}")
    logger.info("=" * 70)
    
    tasks, input_size, output_size = get_benchmark(args.benchmark, args.num_tasks)
    model = SimpleModel(input_size, args.hidden_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    accuracies = []

    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        logger.info(f"\nTASK {task_id}/{args.num_tasks - 1}")
        best_acc = 0
        
        for epoch in range(args.epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                if x.dtype != torch.float32:
                    x = x.float()
                if y.dtype != torch.long:
                    y = y.long()
                logits = model(x)
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    if x.dtype != torch.float32:
                        x = x.float()
                    if y.dtype != torch.long:
                        y = y.long()
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    val_correct += (pred == y).sum().item()
            val_acc = val_correct / len(val_loader.dataset)

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                logger.info(f"  Epoch {epoch}: Val_Acc={val_acc:.4f}")
            best_acc = max(best_acc, val_acc)
        
        task_accs = []
        for eval_task_id in range(task_id + 1):
            _, _, test_loader_eval = tasks[eval_task_id]
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for x, y in test_loader_eval:
                    x, y = x.to(device), y.to(device)
                    if x.dtype != torch.float32:
                        x = x.float()
                    if y.dtype != torch.long:
                        y = y.long()
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    test_correct += (pred == y).sum().item()
            test_acc = test_correct / len(test_loader_eval.dataset)
            task_accs.append(test_acc)

            if test_acc > 0.7:
                status = "GOOD"
            elif test_acc > 0.5:
                status = "MODERATE"
            else:
                status = "POOR"
            logger.info(f"  {status} Task {eval_task_id}: {test_acc:.4f}")
        
        accuracies.append(task_accs)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    
    acc_list = [list(row) for row in accuracies]
    final_accs = accuracies[-1]
    avg_acc = np.mean(final_accs)
    logger.info(f"Average Accuracy: {avg_acc:.4f}")
    
    if args.num_tasks > 1:
        bwt_vals = []
        for i in range(args.num_tasks - 1):
            forgetting = accuracies[i][i] - accuracies[-1][i]
            bwt_vals.append(forgetting)
        bwt = np.mean(bwt_vals)
        logger.info(f"Backward Transfer (Forgetting): {bwt:.4f}")
        logger.info(f" Task-wise forgetting: {[f'{f:.4f}' for f in bwt_vals]}")
    
    logger.info("\nAccuracy Matrix:")
    for task_id, row in enumerate(acc_list):
        logger.info(f" Task {task_id}: {[f'{acc:.4f}' for acc in row]}")
    
    logger.info("=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='permuted_mnist',
                        choices=['permuted_mnist', 'split_cifar', 'rotated_mnist', 'sequence'])
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train_and_evaluate(args)

if __name__ == '__main__':
    main()
