import numpy as np
from typing import List


class ContinualLearningMetrics:
    """Compute standard CL metrics."""
    
    def __init__(self, accuracies: List[List[float]]):
        self.accuracies = np.array(accuracies)
        self.num_tasks = len(accuracies)
    
    def backward_transfer(self) -> float:
        """
        Measure forgetting.
        BWT = (1/T-1) Σ (ACC_i(i) - ACC_i(T))
        """
        if self.num_tasks < 2:
            return 0.0
        
        bwt = 0.0
        for i in range(self.num_tasks - 1):
            acc_after_task_i = self.accuracies[i, i]
            acc_final = self.accuracies[-1, i]
            bwt += acc_after_task_i - acc_final
        
        return bwt / (self.num_tasks - 1)
    
    def forward_transfer(self) -> float:
        """Measure positive transfer from previous tasks."""
        # Simplified: compare task 1 accuracy after task 0 to baseline
        if self.num_tasks < 2:
            return 0.0
        
        # Would need baseline for true forward transfer
        return 0.0
    
    def average_accuracy(self) -> float:
        """Final average accuracy across all tasks."""
        return float(np.mean(self.accuracies[-1, :]))
    
    def forgetting_per_task(self) -> List[float]:
        """Forgetting for each task."""
        forgetting = []
        for i in range(self.num_tasks - 1):
            acc_after = self.accuracies[i, i]
            acc_final = self.accuracies[-1, i]
            forgetting.append(acc_after - acc_final)
        
        return forgetting
    
    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "="*70)
        print("CONTINUAL LEARNING EVALUATION")
        print("="*70)
        
        print("\nAccuracy Matrix:")
        for i, row in enumerate(self.accuracies):
            print(f"After Task {i}: {[f'{acc:.3f}' for acc in row]}")
        
        print(f"\nAverage Final Accuracy: {self.average_accuracy():.4f}")
        print(f"Backward Transfer (Forgetting): {self.backward_transfer():.4f}")
        
        forgetting = self.forgetting_per_task()
        if forgetting:
            print(f"Forgetting per task: {[f'{f:.3f}' for f in forgetting]}")
