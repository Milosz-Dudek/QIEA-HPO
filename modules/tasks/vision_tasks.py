"""
Vision benchmark task (CIFAR-10) for HPO experiments.
"""
from typing import Any, Optional, Callable
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from loguru import logger


class SimpleCIFAR10CNN(nn.Module):
    """
    Lightweight CNN for CIFAR-10.

    - Two convolutional blocks with batch normalization and max pooling
    - Small fully connected head with dropout
    - Designed to train quickly but reach reasonable accuracy (40–60%+ with a few epochs)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward layer"""
        x = self.features(x)
        x = self.classifier(x)
        return x


def make_cifar10_objective(
        data_dir: str = "./data",
        device: str = "cuda",
        max_epochs: int = 5,
) -> tuple:
    """
    Build an objective function for CIFAR-10 image classification.

    Parameters
    ----------
    data_dir : str
        Directory where CIFAR-10 will be downloaded / loaded.
    device : {"cuda", "cpu"}
        Preferred device. If "cuda" is requested but not available,
        the objective will silently fall back to CPU.
    max_epochs : int
        Number of epochs per trial (upper bound). Kept deliberately small
        to make HPO feasible.

    Returns
    -------
    objective : callable
        Function taking a hyperparameter dict and returning (val_acc, info_dict).
    meta : dict
        Metadata about the task.
    """
    use_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    logger.info(f"[CIFAR10] Using device={use_device}, max_epochs={max_epochs}")

    # Standard CIFAR-10 normalization; light augmentations for train
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    # test_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=(0.4914, 0.4822, 0.4465),
    #             std=(0.2023, 0.1994, 0.2010),
    #         ),
    #     ]
    # )

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    # Subsample the training set to speed up each trial.
    # We keep a fixed validation subset for comparability across trials.
    total_len = len(full_train)  # 50_000
    train_len = 20_000
    val_len = 5_000
    unused_len = total_len - train_len - val_len
    if unused_len < 0:
        raise ValueError("Requested train/val sizes exceed CIFAR-10 training set size.")

    train_ds, val_ds, _ = random_split(
        full_train,
        [train_len, val_len, unused_len],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(
        f"[CIFAR10] Dataset split: train={len(train_ds)}, val={len(val_ds)}, "
        f"unused={unused_len}"
    )

    def objective(params: dict[str, Any]):
        """
        CIFAR-10 objective.

        Parameters
        ----------
        params : dict
            Hyperparameters: lr, batch_size, weight_decay.

        Returns
        -------
        val_acc : float
            Validation accuracy in [0, 1].
        info : dict
            Extra info, currently {"time": elapsed_seconds}.
        """
        lr = params["lr"]
        batch_size = int(params["batch_size"])
        weight_decay = params["weight_decay"]

        logger.info(
            f"[CIFAR10] Trial start: lr={lr:.2e}, weight_decay={weight_decay:.2e}, "
            f"batch_size={batch_size}, epochs={max_epochs}"
        )

        model = SimpleCIFAR10CNN(num_classes=10).to(use_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=use_device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=use_device.type == "cuda",
        )

        start = time.time()

        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(use_device), yb.to(use_device)

                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct_train += (preds == yb).sum().item()
                total_train += yb.size(0)

            epoch_loss = running_loss / max(total_train, 1)
            epoch_acc = correct_train / max(total_train, 1)
            logger.info(
                f"[CIFAR10] Epoch {epoch + 1}/{max_epochs}: "
                f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}"
            )

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(use_device), yb.to(use_device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / max(total, 1)
        elapsed = time.time() - start

        logger.info(
            f"[CIFAR10] Finished training. val_acc={acc:.4f}, time={elapsed:.1f}s"
        )

        return acc, {"time": elapsed}

    return objective, {"dataset": "cifar10", "model": "cnn_cifar10"}


class SimpleMNISTCNN(nn.Module):
    """
    Configurable CNN for MNIST.

    The architecture is:
        [Conv2d(+BN) -> ReLU -> MaxPool2d] x num_conv_blocks
        -> Flatten -> Linear(fc_dim) -> ReLU -> Dropout -> Linear(num_classes)
    """

    def __init__(
            self,
            num_classes: int = 10,
            in_channels: int = 1,
            base_channels: int = 32,
            num_conv_blocks: int = 2,
            fc_dim: int = 128,
            use_batchnorm: bool = True,
            dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        channels = in_channels
        for b in range(num_conv_blocks):
            out_ch = base_channels * (2 ** b)
            layers.append(nn.Conv2d(channels, out_ch, kernel_size=3, padding=1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2))
            channels = out_ch

        self.features = nn.Sequential(*layers)

        # After num_conv_blocks pools, spatial size is 28 / 2^num_conv_blocks (floor)
        spatial_dim = 28 // (2 ** num_conv_blocks)
        feat_dim = channels * spatial_dim * spatial_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        """Forward layer"""
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_mnist_loaders(
        batch_size: int = 64,
        num_workers: int = 2,
        valid_ratio: float = 0.2,
        max_train_samples: int = 10000,
        max_val_samples: int = 2000,
):
    """
    Build train/validation DataLoaders for MNIST with an optional subset
    (to keep HPO runs cheap).

    Parameters
    ----------
    batch_size : int
        Batch size for both loaders.
    num_workers : int
        Number of workers for DataLoader.
    valid_ratio : float
        Fraction of the original *train* set reserved for validation
        before subsetting.
    max_train_samples : int
        Upper bound on the number of training samples used for HPO.
    max_val_samples : int
        Upper bound on the number of validation samples used for HPO.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Normalize to mean 0.1307, std 0.3081 (standard MNIST stats)
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Full train set (we'll split into train/val)
    full_train = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    n_total = len(full_train)
    n_val = int(n_total * valid_ratio)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_train,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Optional subsetting to keep HPO fast
    if max_train_samples is not None and max_train_samples < len(train_ds):
        train_ds = torch.utils.data.Subset(
            train_ds, list(range(max_train_samples))
        )
    if max_val_samples is not None and max_val_samples < len(val_ds):
        val_ds = torch.utils.data.Subset(
            val_ds, list(range(max_val_samples))
        )

    logger.info(
        f"[MNIST] Dataset split: train={len(train_ds)}, val={len(val_ds)} "
        f"(original train={n_total}, valid_ratio={valid_ratio})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def make_mnist_objective(
        max_epochs: int = 10,
        device: Optional[str] = None,
) -> tuple[Callable[[dict[str, Any]], tuple[float, dict[str, Any]]], dict[str, Any]]:
    """
    Construct an objective function that trains a configurable CNN on MNIST.

    Returns
    -------
    objective : callable
        params -> (validation_accuracy, info_dict)
        info_dict contains at least {"time": <seconds>, "epochs": int, "acc": float}
    meta : dict
        {"dataset": "mnist", "model": "cnn_mnist"}
    """
    use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for MNIST: {use_device}")

    # This call is mainly to download / cache the data once; loaders are rebuilt per trial.
    _train_loader, _val_loader = get_mnist_loaders(
        batch_size=128,
        num_workers=2,
    )

    def objective(params: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        """Set up the objective in inner function."""
        start = time.time()

        # Architecture hyperparameters
        num_blocks = int(params.get("num_conv_blocks", 2))
        base_channels = int(params.get("base_channels", 32))
        fc_dim = int(params.get("fc_dim", 128))
        use_bn = bool(params.get("use_batchnorm", True))
        dropout = float(params.get("dropout", 0.3))

        # Training hyperparameters
        optimizer_type = params.get("optimizer_type", "adam")
        lr = float(params.get("lr", 1e-3))
        weight_decay = float(params.get("weight_decay", 1e-4))
        batch_size = int(params.get("batch_size", 64))
        epochs = int(params.get("epochs", 5))
        epochs = max(1, min(epochs, max_epochs))

        # Rebuild loaders with trial-specific batch size
        train_loader, val_loader_local = get_mnist_loaders(
            batch_size=batch_size,
            num_workers=2,
        )

        model = SimpleMNISTCNN(
            num_classes=10,
            in_channels=1,
            base_channels=base_channels,
            num_conv_blocks=num_blocks,
            fc_dim=fc_dim,
            use_batchnorm=use_bn,
            dropout=dropout,
        ).to(use_device)

        criterion = nn.CrossEntropyLoss()
        if optimizer_type == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Training loop
        model.train()
        for epoch in range(epochs):
            for xb, yb in train_loader:
                xb = xb.to(use_device)
                yb = yb.to(use_device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader_local:
                xb = xb.to(use_device)
                yb = yb.to(use_device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / max(1, total)
        elapsed = time.time() - start

        info = {
            "time": elapsed,
            "acc": acc,
            "epochs": epochs,
        }
        return acc, info

    meta = {"dataset": "mnist", "model": "cnn_mnist"}
    return objective, meta
