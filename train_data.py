"""
train_data.py — Train dataset and dataloader for CIFAR100.

We use the full CIFAR100 train split (50,000 images, 500 per class) with
shuffle=True. Although the optimiser only sees ``n_batches × batch_size``
samples, sampling them uniformly from the full balanced train set gives
each step a fresh batch with diverse class composition, which improves
the quality of the SPSA pseudo-gradient relative to a fixed subset.

The seeded ``generator_train`` (set in validate.py) makes the batch
sequence reproducible across runs.
"""

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

from augmentation import get_transforms

def get_train_dataset_loader(
    data_dir,
    batch_size,
    generator_train,
):
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
    )
    return train_dataset, train_loader
