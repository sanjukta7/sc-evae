# ExpressionVAE: Elucidating the Design Space of Generative Models for Single-Cell Perturbation Prediction

Sanjukta Bhattacharya, Christian Gensbigler, Shaamil Karim, Jon Lees

[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue?style=for-the-badge)](#citation)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.06.15.732063-b31b1b.svg?style=for-the-badge)](https://www.biorxiv.org/content/10.64898/2026.06.15.732063v1.abstract)
[![PDF](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](#citation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)](LICENSE)

[**Installation**](#installation) | [**Dataset Download**](#dataset-download) | [**Training**](#training) | [**Evaluation**](#evaluation) | [**Citation**](#citation)

<hr style="border: 2px solid gray;"></hr>

`ExpressionVAE` is a vector-quantized (FSQ) variational autoencoder that encodes
each cell's gene-expression profile as a fixed-length sequence of discrete codes,
paired with a perturbation-conditioned generative prior over those codes
(autoregressive, masked discrete diffusion, or flow matching). On Replogle and
Parse 1M it reaches state-of-the-art on the distributional and cell-eval
perturbation metrics, and its frozen encoder transfers to an out-of-distribution
CRISPRi reversion benchmark (TeloHAEC).

This repository contains everything needed to reproduce the paper's two datasets
(Replogle, Parse 1M), the TeloHAEC evaluation, and the reported metrics.

## Installation

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this-repo-url> sc_evae
cd sc_evae
uv sync                 # runtime deps
uv sync --extra dev     # + pytest / black / flake8
```


The scLDM train/holdout split definitions (Replogle HepG2 = 372 perts; Parse
CD4-Naive = 27 cytokines) are JSON specs in `assets/` —
`assets/scldm_replogle_test.json` and `assets/scldm_parse1m_test.json`, each of
shape `{"cell_type": [...], "pert": [...]}`. A config selects one via
`dataset.split_path:`; point it at any JSON of the same shape to run a different
cell-type × perturbation holdout (no code change).

## Training

Training is two-stage.

**Stage 1 — train the `ExpressionVAE`** (Replogle example):

```bash
uv run accelerate launch scripts/train.py \
    --config_path config/final/vaes/reploall-vae-fsq-mse.yaml
```

This writes an experiment directory under `outputs/` containing `config.yaml`,
`dataset_stats.json`, and `checkpoints/`.

**Stage 2 — freeze the VAE and train a prior** over its code sequence:

```bash
uv run accelerate launch scripts/train.py \
    --config_path config/final/prior/reploall-autoregressive-fsq-mse.yaml
```

The prior config references the trained VAE via `model.vae_path:`. Point it at
your Stage-1 run directory (set `prefix_run_name_with_datetime: false` on the VAE
config, or edit `vae_path` to match the actual run-dir name).

### Config layout (`config/final/`)

- `vaes/`     — Stage-1 `ExpressionVAE` configs. `{dataset}-vae-{fsq|gaus}-{head}.yaml`
- `prior/`    — Stage-2 priors. `{dataset}-{autoregressive|mdlm|flow}-{fsq|gaus}-{head}.yaml`
- `nolatent/` — ablation priors trained directly on counts (no VAE latent)

Datasets are `reploall` (Replogle) and `parse1m`. Heads are `mse`, `nb`,
`hurdle`, `ce-quantile`. Bottlenecks are `fsq` (discrete) and `gaus` (Gaussian,
used with flow and in the TeloHAEC grid).

## Evaluation

**Per-perturbation + distribution metrics** (reads `{train_dir}/config.yaml`):

```bash
uv run python scripts/eval.py --train_dir outputs/<your_prior_run>
```

This computes three metric families, written to
`{train_dir}/eval_results/summary_metrics.json`:

- `distribution` — W₂, MMD², Fréchet distance (`metrics/distribution.py`)
- `cell_eval`    — disc_l1, PR-AUC, Spearman-sig/LFC, overlap@N, Pearson-Δ
  (`metrics/state_metrics.py`) — **the paper's reported per-perturbation suite**

**TeloHAEC reversion benchmark** (frozen encoder):

```bash
uv run python scripts/eval_pfizer.py \
    --train_dir outputs/<your_vae_run> \
    --h5ad_path datasets/pfizer/pfizer_raw.h5ad \
    --gene_order_path assets/replogle_scldm2k_gene_order.txt --num_hvg 2000
```

The raw file is sliced to the model's 2000-gene panel on the fly by the
data loader (`gene_order_path` permutes + zero-fills; `num_hvg` slices) — no
separate alignment step. Reports per-perturbation Calinski-Harabasz separability
and the NF-κB enrichment AUC (`metrics/pfizer_ranking.py`).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{Bhattacharya2026,
    author    = {Bhattacharya, Sanjukta and Gensbigler, Christian and Karim, Shaamil and Lees, Jon},
    title     = {Elucidating the Design Space of Generative Models for Single-Cell Perturbation Prediction},
    year      = {2026},
    doi       = {10.64898/2026.06.15.732063},
    publisher = {Cold Spring Harbor Laboratory},
    journal   = {bioRxiv},
    url       = {https://www.biorxiv.org/content/10.64898/2026.06.15.732063v1}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.

## Acknowledgements

This repository builds heavily off of [STATE](https://github.com/ArcInstitute/state),
[scLDM](https://github.com/czi-ai/scLDM), [DiT](https://github.com/facebookresearch/DiT),
and [Score-Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
We also used the [cell-load](https://github.com/ArcInstitute/cell-load) package
introduced in the STATE repository.
