"""
zo_optimizer.py — Zero-order optimizer using multi-query SPSA + momentum.

Strategy
--------
We use **SPSA** (Simultaneous Perturbation Stochastic Approximation): per
step a single random direction is sampled across all active parameters,
two forward passes (at +eps and -eps along that direction) yield a scalar
projected gradient, and the per-parameter pseudo-gradient is the projected
gradient times the random direction.

Per query:

    f_plus  = loss(theta + eps * z)
    f_minus = loss(theta - eps * z)
    proj_grad = (f_plus - f_minus) / (2 * eps)
    grad_estimate[name] = proj_grad * z[name]

For finite ``eps``, the estimator approximates the gradient of a Gaussian-
smoothed objective; its bias decreases with ``eps`` while its per-element
variance remains O(||grad||^2). We mitigate this variance with
**multi-query SPSA**: each ``.step()`` averages ``num_queries`` independent
SPSA estimates on the *same* batch, cutting per-step variance by ``1/q``
in exchange for ``2 * q`` forward passes per step. The compute budget is
counted in samples (n_batches * batch_size), so extra forwards do *not*
spend budget — only wall-clock time. Additional momentum SGD smooths the
remaining noise across steps.

Cumulative SNR for the gradient signal scales as sqrt(T * q), where T is
the number of optimiser steps. Going from q=1 to q=4 buys a 2x SNR gain at
4x wall-clock cost.

Layer choice: we restrict tuning to ``fc.weight`` and ``fc.bias``. The
pretrained ResNet18 backbone already produces strong features; SPSA on the
51,300-parameter head is the right scope for the 8k-sample budget.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Multi-query SPSA optimiser with heavy-ball momentum.

    Note:
        ``validate.py`` instantiates this optimiser as
        ``ZeroOrderOptimizer(model)``, so all hyperparameters must be set
        via constructor defaults below.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 8e-4,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
        momentum: float = 0.9,
        num_queries: int = 1,
        weight_decay: float = 0.0,
    ) -> None:
        self.model = model
        self.lr = float(lr)
        self.eps = float(eps)

        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode
        self.momentum = float(momentum)
        self.num_queries = int(num_queries)
        self.weight_decay = float(weight_decay)

        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if self.num_queries < 1:
            raise ValueError(f"num_queries must be at least 1, got {num_queries}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        self.layer_names: List[str] = ["fc.weight", "fc.bias"]

        self._velocity: Dict[str, torch.Tensor] = {}
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> Dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample one IID perturbation tensor matching ``param``'s shape."""
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(param)
        # "uniform" -> Rademacher (±1)
        u = torch.empty_like(param).bernoulli_(0.5)
        return u.mul_(2.0).sub_(1.0)

    def _spsa_query(
        self,
        loss_fn: Callable[[], float],
        params: Dict[str, nn.Parameter],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """One central-difference SPSA query. 2 forward passes."""
        directions = {n: self._sample_direction(p) for n, p in params.items()}

        offset = 0.0
        try:
            with torch.no_grad():
                for n, p in params.items():
                    p.add_(directions[n], alpha=self.eps)
            offset = self.eps
            f_plus = loss_fn()

            with torch.no_grad():
                for n, p in params.items():
                    p.add_(directions[n], alpha=-2.0 * self.eps)
            offset = -self.eps
            f_minus = loss_fn()
        finally:
            if offset:
                with torch.no_grad():
                    for n, p in params.items():
                        p.add_(directions[n], alpha=-offset)

        proj_grad = (f_plus - f_minus) / (2.0 * self.eps)
        grads = {n: directions[n] * proj_grad for n in params}
        return grads, 0.5 * (f_plus + f_minus)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: Dict[str, nn.Parameter],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Multi-query SPSA: average ``num_queries`` independent SPSA estimates.

        Costs ``2 * num_queries`` forward passes per call. Each query uses a
        fresh random direction but the *same* batch (loss_fn closes over a
        fixed batch in validate.py).
        """
        if self.num_queries <= 1:
            return self._spsa_query(loss_fn, params)

        accum = {n: torch.zeros_like(p) for n, p in params.items()}
        loss_sum = 0.0
        for _ in range(self.num_queries):
            grads_q, loss_q = self._spsa_query(loss_fn, params)
            for n in accum:
                accum[n].add_(grads_q[n])
            loss_sum += loss_q

        inv_q = 1.0 / self.num_queries
        for n in accum:
            accum[n].mul_(inv_q)
        return accum, loss_sum * inv_q

    def _update_params(
        self,
        params: Dict[str, nn.Parameter],
        grads: Dict[str, torch.Tensor],
    ) -> None:
        """Heavy-ball momentum SGD: v <- beta*v + (1-beta)*g; theta <- theta - lr*v."""
        with torch.no_grad():
            for n, p in params.items():
                g = grads[n]
                if self.weight_decay > 0.0:
                    g = g.add(p, alpha=self.weight_decay)

                if n not in self._velocity:
                    self._velocity[n] = torch.zeros_like(p)
                v = self._velocity[n]
                v.mul_(self.momentum).add_(g, alpha=1.0 - self.momentum)
                p.add_(v, alpha=-self.lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one optimiser step. Calls ``loss_fn`` ``2 * num_queries`` times."""
        params = self._active_params()
        grads, loss_estimate = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        self._step_count += 1
        return float(loss_estimate)
