# Lensless Computational Imaging

DigiCam lensless reconstruction homework. I implemented ADMM-100, LeADMM-20, and modular LeADMM-5 + DRUNet (pre / post / pre+post). Best test result I got: **17.31 dB PSNR** with pre+post DRUNet (`outputs/final_leadmm5_drunet8m_prepost`). Also added **FlatNet-lite** as a bonus — learned Wiener inversion + DRUNet.

Configs live in `src/configs/`. Metrics use ROI crop `[80:280, 100:366]`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_digicam_masks.py   # mask_0.npy … mask_99.npy into data/digicam_masks/
```

Dataset pulls from HuggingFace on first run. For Comet logging during training, I put my API key in `private_tokens.py` (gitignored).

```bash
python download_checkpoints.py --name leadmm5_prepost_drunet8m.pth   # demo / best model
python download_checkpoints.py --all                               # all report checkpoints
```

Checkpoints live on HF: `tolyho/lensless-computational-imaging-checkpoints`.

## Inference

Default `-cn inference` is fixed ADMM-100 + background subtraction (no checkpoint).

```bash
.venv/bin/python inference.py -cn inference

.venv/bin/python inference.py -cn inference_leadmm20 \
  checkpoint=outputs/train_leadmm20/checkpoint-epochN.pth \
  output_dir=outputs/digicam_leadmm20

.venv/bin/python inference.py -cn inference_modular_prepost \
  model=modular_leadmm5_prepost_drunet8m \
  checkpoint=outputs/train_leadmm5_drunet8m_prepost_ep10/leadmm5_drunet8m_prepost_ep10/model_best.pth \
  output_dir=outputs/final_leadmm5_drunet8m_prepost
```

## Metrics

Compare `recon/` vs `lensed/` from an inference run:

```bash
.venv/bin/python calculate_metrics.py \
  pred_dir=outputs/digicam_simple_admm100/recon \
  gt_dir=outputs/digicam_simple_admm100/lensed \
  output_path=outputs/digicam_simple_admm100/metrics.json
```

## Training

Example: train LeADMM-20. I logged runs to Comet via `-cn train_comet`.

```bash
.venv/bin/python train.py -cn train_comet \
  model.num_iters=20 model.cg_iters=10 \
  datasets.train.limit=null datasets.val.limit=null \
  trainer.n_epochs=6 \
  trainer.save_dir=outputs/train_leadmm20 \
  writer.run_name=leadmm20 \
  optimizer.lr=1e-3 \
  trainer.max_grad_norm=100 \
  trainer.log_step=50
```

## Modular Models

LeADMM-5 unrolled core with DRUNet before/after. Pre+post worked best for me.

```bash
.venv/bin/python train.py -cn train_comet \
  model=modular_leadmm5_prepost_drunet8m \
  trainer.n_epochs=10 \
  trainer.save_dir=outputs/train_leadmm5_drunet8m_prepost_ep10 \
  writer.run_name=leadmm5_drunet8m_prepost_ep10
```

## FlatNet-Lite

Bonus baseline: I learn only Wiener λ/scale/bias, then DRUNet cleans up. Debug run first, then full training:

```bash
# debug
.venv/bin/python train.py -cn train_comet \
  model=flatnet_lite \
  datasets.train.limit=20 datasets.val.limit=5 \
  trainer.n_epochs=1 optimizer.lr=1e-4 \
  trainer.save_dir=outputs/train_flatnet_lite_debug \
  writer.run_name=flatnet_lite_debug

# full (10 epochs)
.venv/bin/python train.py -cn train_comet \
  model=flatnet_lite \
  datasets.train.limit=null datasets.val.limit=null \
  trainer.n_epochs=10 optimizer.lr=1e-4 \
  trainer.save_dir=outputs/train_flatnet_lite \
  writer.run_name=flatnet_lite

# follow training log
tail -f outputs/train_flatnet_lite/flatnet_lite/info.log

.venv/bin/python inference.py -cn inference_flatnet \
  checkpoint=outputs/train_flatnet_lite/flatnet_lite/checkpoint-epoch10.pth \
  output_dir=outputs/final_flatnet_lite
```

## Custom Data

Run on your own folder layout:

```bash
.venv/bin/python inference.py -cn inference_custom \
  dataset.root=/path/to/data \
  output_dir=outputs/custom_run
```
