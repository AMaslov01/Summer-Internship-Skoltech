"""
head_init.py — Final layer initialization for ZO fine-tuning.

We use a small-scale Gaussian init for the new 100-class head: weights drawn
from N(0, 0.01^2), bias zero-initialised. Rationale:

* With small weights the initial logits are near zero and the softmax is
  approximately uniform, so the initial cross-entropy loss is close to the
  ideal log(K) = log(100) ~ 4.605. This is the "neutral" starting point a
  ZO method can improve from in any direction.
* Larger initialisations (Kaiming, Xavier) push the softmax toward
  saturated regimes where most of the loss surface is flat, which is
  catastrophic for finite-difference estimators — f(theta+eps*z) and
  f(theta-eps*z) become indistinguishable.
* Zero init for the bias keeps the per-class prior uniform until the
  optimizer learns otherwise.
"""

import torch.nn as nn

# Small Gaussian std keeps initial logits near zero -> softmax near uniform ->
# loss near log(100). Empirically the most stable init for ZO fine-tuning.
_INIT_STD = 0.01


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the new CIFAR100 head in-place."""
    nn.init.normal_(layer.weight, mean=0.0, std=_INIT_STD)
    nn.init.zeros_(layer.bias)
