"""Model definition, training, testing, and data loading for CIFAR-10."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Works across:
    - NVIDIA GPUs (CUDA) - Linux x86
    - Apple Silicon (MPS) - MacBook M4
    - CPU fallback - Raspberry Pi and others
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Net(nn.Module):
    """Simple CNN for CIFAR-10 classification.

    Architecture adapted from PyTorch's '60 Minute Blitz' tutorial.
    Input: 3x32x32 (CIFAR-10 images)
    Output: 10 classes
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Load and partition CIFAR-10 dataset.

    Uses Flower Datasets to download CIFAR-10 and partition it using IID partitioning.
    Each client gets a unique partition based on partition_id.

    Args:
        partition_id: The ID of this client's partition (0 to num_partitions-1)
        num_partitions: Total number of partitions across all clients
        batch_size: Batch size for DataLoaders

    Returns:
        Tuple of (train_loader, test_loader) for this partition
    """
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    partition = fds.load_partition(partition_id)

    # Split partition: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # CIFAR-10 normalization
    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    def apply_transforms(batch: dict) -> dict:
        """Apply image transforms to batch."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility with Raspberry Pi
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, testloader


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train the model on local data.

    Args:
        net: The neural network model
        trainloader: DataLoader for training data
        epochs: Number of local training epochs
        lr: Learning rate
        device: Device to train on (cuda/mps/cpu)

    Returns:
        Average training loss over all batches
    """
    net.to(device)
    net.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on test data.

    Args:
        net: The neural network model
        testloader: DataLoader for test data
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            total_loss += criterion(outputs, labels).item() * len(labels)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, accuracy
