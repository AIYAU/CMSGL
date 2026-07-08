# CoSGL

Official code release for **CoSGL: Class-Conditioned Spectral-Geometric Learning for Open-Set HSI-LiDAR Classification**.

This repository contains only the proposed CoSGL / S2G-CCSGCL method implementation. Comparison-method adapters, external baseline repositories, manuscript drafts, datasets, trained weights, and generated experiment outputs are intentionally excluded.

## Repository Layout

```text
CoSGL/
|-- cc_sgcl/        # Proposed model, data pipeline, training, inference, metrics
|-- configs/        # Reproducibility configs
|-- scripts/        # Training, evaluation, calibration, and summary entry points
|-- third_party/    # Small compatibility helpers used by the data protocol
|-- data/           # Placeholder only; datasets are not included
|-- docs/           # Release notes
|-- requirements.txt
|-- pyproject.toml
`-- README.md
```

## Installation

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data

Datasets are not included. Place prepared arrays under `data/`.

For local `.mat` files, a helper script is provided:

```powershell
python scripts/prepare_local_mat_data.py --dataset Trento --unknown_classes 6
```

The paper experiments use a HyLiOSR-compatible known/unknown open-set split protocol.

## Quick Smoke Test

```powershell
python scripts/run_cc_sgcl.py --config configs/final_s2g.yaml --data_root data --data 1 --epochs 1 --batch_size 16 --numTrain 2 --device cpu --save_model false --save_maps false --output_dir outputs/smoke
```

## Main Entry Point

```powershell
python scripts/run_cc_sgcl.py --config configs/final_s2g.yaml --data_root data --data 1
```

Use `--data 1`, `--data 2`, and `--data 3` for the three paper datasets after preparing the corresponding data files.

## Notes

- Do not commit datasets, checkpoints, generated maps, or experiment outputs.
- The optional Vision-RWKV code path is retained for experimental variants. Set `COSGL_VISION_RWKV_ROOT` if you use it; the main released config uses the CNN encoder.
- Add citation information here after the manuscript DOI or preprint link is available.
