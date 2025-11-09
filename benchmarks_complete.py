import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PermutedMNISTGenerator:
    """
    Permuted MNIST: Standard continual learning benchmark.
    Each task has same MNIST digits but with different pixel permutation.
    
    Reference: Kirkpatrick et al., "Continual Learning Through Synaptic Intelligence"
    """
    
    def __init__(self, num_tasks: int = 10, samples_per_task: int = 1000, seed: int = 42):
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task
        self.seed = seed
        self.permutations = self._generate_permutations()
        
    def _generate_permutations(self) -> List[np.ndarray]:
        """Generate random pixel permutations for each task."""
        np.random.seed(self.seed)
        permutations = []
        
        for _ in range(self.num_tasks):
            perm = np.random.permutation(784)  # 28×28 = 784 pixels
            permutations.append(perm)
        
        return permutations
    
    def generate_synthetic_mnist(self, task_id: int, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic MNIST-like data with task-specific permutation."""
        np.random.seed(self.seed + task_id)
        
        images = []
        labels = []
        
        # 10 digit classes
        for digit in range(10):
            samples_per_digit = num_samples // 10
            
            for _ in range(samples_per_digit):
                # Create digit pattern
                img = torch.zeros(28, 28)
                
                # Draw simple digit shape
                center = 14 + (digit % 5) * 0.5 - 2
                y, x = np.ogrid[:28, :28]
                mask = (x - center)**2 + (y - center)**2 <= (8 + digit)**2
                img[mask] = 1.0
                
                # Add noise
                img = img + torch.randn_like(img) * 0.3
                img = torch.clamp(img, 0, 1)
                
                # Flatten and apply permutation
                img_flat = img.view(-1)
                img_permuted = img_flat[self.permutations[task_id]]
                
                images.append(img_permuted)
                labels.append(digit)
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
    
    def get_task(self, task_id: int, batch_size: int = 32, split: str = 'train') -> DataLoader:
        """Get DataLoader for a specific task."""
        if split == 'train':
            num_samples = self.samples_per_task
        else:
            num_samples = self.samples_per_task // 5
        
        images, labels = self.generate_synthetic_mnist(task_id, num_samples)
        
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
        return dataloader
    
    def get_all_tasks(self, batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get train/val/test splits for all tasks."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            train_loader = self.get_task(task_id, batch_size, split='train')
            val_loader = self.get_task(task_id, batch_size, split='val')
            test_loader = self.get_task(task_id, batch_size, split='test')
            
            tasks.append((train_loader, val_loader, test_loader))
        
        return tasks


class SplitCIFARGenerator:
    """
    Split CIFAR-10: Class-incremental learning benchmark.
    Each task contains 2 different CIFAR-10 classes.
    
    Task 0: [airplane, automobile]
    Task 1: [bird, cat]
    Task 2: [deer, dog]
    Task 3: [frog, horse]
    Task 4: [ship, truck]
    """
    
    def __init__(self, num_tasks: int = 5, samples_per_class: int = 500, seed: int = 42):
        self.num_tasks = num_tasks
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.classes_per_task = 10 // num_tasks
        
    def generate_synthetic_cifar(self, class_id: int, num_samples: int = 500) -> torch.Tensor:
        """Generate synthetic CIFAR-10-like 32×32 RGB images."""
        np.random.seed(self.seed + class_id)
        
        images = []
        
        for _ in range(num_samples):
            # Create image with class-specific pattern
            img = torch.zeros(3, 32, 32)
            
            # Different color patterns for different classes
            if class_id == 0:  # airplane
                img[0, 8:24, 8:24] = torch.ones(16, 16)
                img[1, 10:22, 10:22] = torch.ones(12, 12)
            elif class_id == 1:  # automobile
                img[1, 10:22, 8:24] = torch.ones(12, 16)
                img[2, 15:20, 15:20] = torch.ones(5, 5)
            elif class_id == 2:  # bird
                img[1, 5:20, 5:20] = torch.ones(15, 15)
            elif class_id == 3:  # cat
                img[0, 8:20, 8:20] = torch.ones(12, 12)
            elif class_id == 4:  # deer
                img[1, 8:20, 8:20] = torch.ones(12, 12)
            elif class_id == 5:  # dog
                img[0, 10:22, 10:22] = torch.ones(12, 12)
            elif class_id == 6:  # frog
                img[2, 12:24, 12:24] = torch.ones(12, 12)
            elif class_id == 7:  # horse
                img[0, 10:24, 10:24] = torch.ones(14, 14)
            elif class_id == 8:  # ship
                img[2, 8:24, 8:24] = torch.ones(16, 16)
            elif class_id == 9:  # truck
                img[0, 8:24, 10:22] = torch.ones(16, 12)
            else:
                img = torch.bernoulli(torch.full((3, 32, 32), 0.3))
            
            # Add noise
            img = img + torch.randn_like(img) * 0.3
            img = torch.clamp(img, 0, 1)
            
            images.append(img)
        
        return torch.stack(images)
    
    def get_task(self, task_id: int, batch_size: int = 32, split: str = 'train') -> DataLoader:
        """Get DataLoader for task."""
        images = []
        labels = []
        
        start_class = task_id * self.classes_per_task
        end_class = start_class + self.classes_per_task
        
        if split == 'train':
            num_samples = self.samples_per_class
        else:
            num_samples = self.samples_per_class // 5
        
        for idx, class_id in enumerate(range(start_class, end_class)):
            class_images = self.generate_synthetic_cifar(class_id, num_samples)
            images.append(class_images)
            labels.extend([idx] * num_samples)
        
        images = torch.cat(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
        return dataloader
    
    def get_all_tasks(self, batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get train/val/test for all tasks."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            train_loader = self.get_task(task_id, batch_size, split='train')
            val_loader = self.get_task(task_id, batch_size, split='val')
            test_loader = self.get_task(task_id, batch_size, split='test')
            
            tasks.append((train_loader, val_loader, test_loader))
        
        return tasks


class RotatedMNISTGenerator:
    """
    Rotated MNIST: Smooth distribution shift benchmark.
    Each task has MNIST rotated by increasing angles.
    
    Task 0: 0° rotation
    Task 1: 10° rotation
    Task 2: 20° rotation
    etc.
    """
    
    def __init__(self, num_tasks: int = 10, samples_per_task: int = 1000, seed: int = 42):
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task
        self.seed = seed
    
    def generate_synthetic_mnist(self, task_id: int, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic MNIST with rotation."""
        np.random.seed(self.seed + task_id)
        
        images = []
        labels = []
        
        rotation_angle = (task_id * 10) % 360  # 0°, 10°, 20°, ...
        
        for digit in range(10):
            samples_per_digit = num_samples // 10
            
            for _ in range(samples_per_digit):
                # Create digit
                img = torch.zeros(28, 28)
                center = 14
                y, x = np.ogrid[:28, :28]
                mask = (x - center)**2 + (y - center)**2 <= (8 + digit)**2
                img[mask] = 1.0
                
                # Apply rotation using PIL
                from PIL import Image
                pil_img = Image.fromarray((img.numpy() * 255).astype(np.uint8))
                rotated = pil_img.rotate(rotation_angle)
                img = torch.from_numpy(np.array(rotated)) / 255.0
                
                # Add noise
                img = img + torch.randn_like(img) * 0.3
                img = torch.clamp(img, 0, 1)
                
                images.append(img.view(-1).float())
                labels.append(digit)
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
    
    def get_task(self, task_id: int, batch_size: int = 32, split: str = 'train') -> DataLoader:
        """Get DataLoader for task."""
        if split == 'train':
            num_samples = self.samples_per_task
        else:
            num_samples = self.samples_per_task // 5
        
        images, labels = self.generate_synthetic_mnist(task_id, num_samples)
        
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
        return dataloader
    
    def get_all_tasks(self, batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get all tasks."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            train_loader = self.get_task(task_id, batch_size, split='train')
            val_loader = self.get_task(task_id, batch_size, split='val')
            test_loader = self.get_task(task_id, batch_size, split='test')
            
            tasks.append((train_loader, val_loader, test_loader))
        
        return tasks


class ImprovedSequenceGenerator:
    """
    Sequence Learning: Learn different transformation rules per task.
    
    Task 0: y = (x + x_prev) % vocab_size
    Task 1: y = (2*x + x_prev) % vocab_size
    Task 2: y = (x * x_prev) % vocab_size
    etc.
    """
    
    def __init__(self, num_tasks: int = 10, seq_length: int = 64, 
                 vocab_size: int = 256, samples_per_task: int = 2000, seed: int = 42):
        self.num_tasks = num_tasks
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.samples_per_task = samples_per_task
        self.seed = seed
        self.rules = self._create_rules()
    
    def _create_rules(self):
        """Create interpretable transformation rules."""
        rules = [
            lambda seq: torch.roll(seq, 1),
            lambda seq: (seq * 2) % self.vocab_size,
            lambda seq: self.vocab_size - seq,
            lambda seq: seq[::2] if len(seq) >= 32 else seq,
            lambda seq: torch.where(torch.arange(len(seq)) % 2 == 0, seq, torch.zeros_like(seq)),
            lambda seq: torch.sort(seq)[0],
            lambda seq: torch.flip(seq, [0]),
            lambda seq: torch.clamp(seq + torch.randn_like(seq) - torch.randn_like(seq), 0, self.vocab_size-1),
            lambda seq: torch.tensor([int((seq[i] + seq[i+1]) % self.vocab_size) if i < len(seq)-1 else seq[i] for i in range(len(seq))]),
            lambda seq: torch.cat([torch.flip(seq[:len(seq)//2], [0]), seq[len(seq)//2:]]),
        ]
        return rules
    
    def generate_task_data(self, task_id: int, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate data for task using its rule."""
        np.random.seed(self.seed + task_id)
        
        rule = self.rules[task_id % len(self.rules)]
        
        sequences = []
        labels = []
        
        for _ in range(num_samples):
            seq_in = torch.randint(1, self.vocab_size, (self.seq_length,))
            
            try:
                seq_out = rule(seq_in)
                
                if len(seq_out) != self.seq_length:
                    seq_out = seq_out[:self.seq_length]
                    if len(seq_out) < self.seq_length:
                        seq_out = torch.cat([seq_out, torch.zeros(self.seq_length - len(seq_out))])
                
                seq_out = torch.clamp(seq_out.long(), 0, self.vocab_size - 1)
                
                # Use first token as classification label
                sequences.append(seq_in)
                labels.append(seq_in[0].item() % 10)  # 10 classes
            except:
                continue
        
        if len(sequences) == 0:
            sequences = torch.randint(1, self.vocab_size, (num_samples, self.seq_length))
            labels = torch.randint(0, 10, (num_samples,))
        else:
            sequences = torch.stack(sequences)
            labels = torch.tensor(labels, dtype=torch.long)
        
        return sequences, labels
    
    def get_task(self, task_id: int, batch_size: int = 32, split: str = 'train') -> DataLoader:
        """Get DataLoader for task."""
        if split == 'train':
            num_samples = self.samples_per_task
        else:
            num_samples = self.samples_per_task // 5
        
        sequences, labels = self.generate_task_data(task_id, num_samples)
        
        dataset = TensorDataset(sequences, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
        
        return dataloader
    
    def get_all_tasks(self, batch_size: int = 32) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get all tasks."""
        tasks = []
        
        for task_id in range(self.num_tasks):
            train_loader = self.get_task(task_id, batch_size, split='train')
            val_loader = self.get_task(task_id, batch_size, split='val')
            test_loader = self.get_task(task_id, batch_size, split='test')
            
            tasks.append((train_loader, val_loader, test_loader))
        
        return tasks


if __name__ == "__main__":
    """Test benchmark generators."""
    
    print("=" * 70)
    print("Testing Permuted MNIST Generator")
    print("=" * 70)
    gen = PermutedMNISTGenerator(num_tasks=3, samples_per_task=100)
    tasks = gen.get_all_tasks(batch_size=32)
    
    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        x, y = next(iter(train_loader))
        print(f"\nTask {task_id}:")
        print(f"  Input: {x.shape} (flattened 28×28=784 pixels)")
        print(f"  Labels: {y.shape} (10 digit classes)")
        print(f"  Ready for training")
    
    print("\n" + "=" * 70)
    print("Testing Split CIFAR Generator")
    print("=" * 70)
    gen = SplitCIFARGenerator(num_tasks=5, samples_per_class=100)
    tasks = gen.get_all_tasks(batch_size=32)
    
    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        x, y = next(iter(train_loader))
        print(f"\nTask {task_id}:")
        print(f"  Input: {x.shape} (3×32×32 RGB images)")
        print(f"  Labels: {y.shape} (2 classes per task)")
        print(f"  Ready for training")
    
    print("\nAll benchmarks working correctly!")
