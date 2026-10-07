import unittest

import torch
import torch.nn as nn

from zo_optimizer import ZeroOrderOptimizer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.fc = nn.Linear(2, 1)


class ZeroOrderOptimizerTests(unittest.TestCase):
    def test_rejects_invalid_hyperparameters(self):
        cases = [
            {"lr": 0.0},
            {"eps": 0.0},
            {"momentum": -0.1},
            {"momentum": 1.0},
            {"num_queries": 0},
            {"weight_decay": -0.1},
            {"perturbation_mode": "invalid"},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ZeroOrderOptimizer(TinyModel(), **kwargs)

    def test_restores_parameters_when_first_loss_evaluation_fails(self):
        model = TinyModel()
        optimizer = ZeroOrderOptimizer(model)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}

        def fail():
            raise RuntimeError("evaluation failed")

        with self.assertRaises(RuntimeError):
            optimizer.step(fail)

        self._assert_parameters_equal(model, before)

    def test_restores_parameters_when_second_loss_evaluation_fails(self):
        model = TinyModel()
        optimizer = ZeroOrderOptimizer(model)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        calls = 0

        def fail_on_second_call():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("evaluation failed")
            return 1.0

        with self.assertRaises(RuntimeError):
            optimizer.step(fail_on_second_call)

        self._assert_parameters_equal(model, before)

    def test_step_updates_only_selected_layers(self):
        torch.manual_seed(7)
        model = TinyModel()
        optimizer = ZeroOrderOptimizer(model, lr=0.01, eps=0.01, momentum=0.0)
        before_fc = model.fc.weight.detach().clone()
        before_backbone = model.backbone.weight.detach().clone()

        def loss():
            return float(
                sum(parameter.detach().square().sum() for parameter in model.fc.parameters())
            )

        value = optimizer.step(loss)

        self.assertIsInstance(value, float)
        self.assertFalse(torch.equal(model.fc.weight, before_fc))
        torch.testing.assert_close(model.backbone.weight, before_backbone, rtol=0.0, atol=0.0)

    def _assert_parameters_equal(self, model, expected):
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, expected[name], rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
