"""
augmentation.py — Training augmentation pipeline for CIFAR100.

We use a deliberately *light* training pipeline. ZO methods are sensitive to
loss noise — anything that makes the per-batch loss vary more across steps
weakens the SPSA gradient signal. Heavy augmentation (AutoAugment,
RandomErasing with high p, large ColorJitter) was tried and consistently
hurt the fine-tuned accuracy in the 8k-sample budget regime.

The augmentations kept:
  * RandomHorizontalFlip               — light label-preserving regularisation
  * RandomCrop with small padding      — translation invariance via shifts of
                                         the upscaled 224x224 image
  * Mild ColorJitter                   — robustness to colour drift; small
                                         magnitudes so loss stays stable

The validation pipeline must remain identical — do not edit.
"""

import torchvision.transforms as T

# Per-channel mean and std computed on the CIFAR100 training set.
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(train: bool) -> T.Compose:
    """Return the image transform pipeline for CIFAR100."""
    if train:
        return T.Compose(
            [
                T.Resize(224),
                T.RandomHorizontalFlip(),
                T.RandomCrop(224, padding=16),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
            ]
        )
    # Fixed validation pipeline — do not modify.
    return T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
        ]
    )
