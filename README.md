# Compute-Constrained Zero-Order Fine-Tuning

An experiment in black-box fine-tuning of a pretrained ResNet18 on CIFAR-100
without gradients. The optimizer can observe scalar loss values only, and the
training budget is limited to **8,192 samples**.

I replaced the original per-parameter finite-difference estimator with SPSA:
all selected parameters are perturbed along one random direction, so the final
configuration needs two forward evaluations per step regardless of parameter
count. Only the 51,300-parameter classification head is optimized.

## Recorded result

| Checkpoint | Top-1 accuracy |
|---|---:|
| ImageNet head baseline | 0.37% |
| New head before fine-tuning | 0.89% |
| SPSA fine-tuned head | **1.50%** |

Final budget: `128 steps × 64 samples = 8,192 samples`.

The original generated `results.json` is not present in this repository. The
values above are recorded in [`SOLUTION.md`](SOLUTION.md); running the command
below regenerates a result file with the current code and environment.

## Method

- two-point SPSA with Gaussian perturbations;
- heavy-ball momentum for variance reduction;
- optimization restricted to `fc.weight` and `fc.bias`;
- small Gaussian initialization for the new head;
- light crop, color, and horizontal-flip augmentation;
- fixed random seeds and deterministic PyTorch settings where supported.

The full experiment log, ablations, failed approaches, and limitations are in
[`SOLUTION.md`](SOLUTION.md).

## Repository layout

```text
zo_optimizer.py   SPSA gradient estimate and momentum update
head_init.py      Classification-head initialization
augmentation.py   Training and evaluation transforms
train_data.py     CIFAR-100 training loader
model.py          Fixed ResNet18 construction
validate.py       Fixed experiment runner
SOLUTION.md       Experiment record and failure analysis
```

## Reproduce

Python 3.13 was used for the final experiment.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python validate.py \
  --data_dir ./data \
  --batch_size 64 \
  --n_batches 128 \
  --output results.json
```

CIFAR-100 is downloaded on the first run. The generated JSON uses these keys:

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.0089,
  "val_accuracy_top1_finetuned": 0.015,
  "n_batches": 128,
  "batch_size": 64,
  "layers_tuned": ["fc.weight", "fc.bias"],
  "total_samples": 10000
}
```

Here, `total_samples` is the size of the validation split. The optimization
budget is reported separately by `n_batches × batch_size` and remains 8,192
training samples in the configuration above.

## Tests

The optimizer tests use a tiny local model and do not download CIFAR-100:

```bash
python -m unittest discover -s tests -v
```

## Limitations

- The final 1.50% accuracy is low in absolute terms; the main result is the
  compute reduction and the measured behavior of SPSA under a hard budget.
- Hyperparameters were selected against the challenge evaluation split, so the
  reported score is a challenge result rather than an untouched test estimate.
- Reproducibility across MPS, CUDA, and CPU can still vary despite fixed seeds.
