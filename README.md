# Lensless Computational Imaging

My homework repo for DigiCam lensless reconstruction — fixed ADMM-100, trainable LeADMM-20, and modular CNN variants (still in progress).

I use [DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K): train split for learning, test split (1500 images) for evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_digicam_masks.py   # mask_0.npy … mask_99.npy into data/digicam_masks/
```

For PSF simulation and DigiCam preprocessing I use the course [`lensless_helpers`](https://github.com/Blinorot/deep-learning-research/tree/2026/hw05/lensless_helpers) — I vendored it at the repo root. Each 54×26 LCD mask goes through `simulate_psf_from_mask` (wave propagation, RPi HQ sensor, Adafruit LCD, downsample=8) and becomes a 380×507 RGB PSF. I also rotate lensless images 180° and align lensed ground truth into the sensor frame, same as the course notebook.

## Fixed ADMM-100 baseline

I implemented 100-iteration ADMM with fixed μ=1e-4, τ=2e-4 and no learned weights. Config: `src/configs/inference.yaml` (`model._target_: src.model.ADMM`).

The raw output sat at ~6.8 dB. Root-cause work (see `docs/code_analysis.md` #10 and `docs/experiments.md` Exp 1) showed this was **not** a fundamental limit: the reconstruction is spatially aligned with the GT, but it reconstructs in sensor-intensity units and leaves a flat grey DC pedestal over the (mostly-black) frame. Removing that pedestal — GT-free per-channel background subtraction (`subtract_background` in `src/model/admm.py`, enabled in the config) — more than doubles PSNR.

**My results on DigiCam test (1500 images):**

| Variant | PSNR | SSIM | MSE | LPIPS |
|---------|------|------|-----|-------|
| raw ADMM-100 | 6.82 | 0.120 | 0.214 | 0.826 |
| **+ background subtraction (q=0.6)** | **14.80** | **0.562** | **0.039** | **0.622** |

Honest caveat: the GT is ~74% black frame, so most of the full-frame jump is from correctly blacking that out; inside the scene ROI the gain is a smaller but real +2.7 dB (to ~10.6 dB). The scene itself is still blurry (LPIPS 0.62) — real detail recovery is the job of the trained LeADMM and modular variants. `q=0.6` assumes the mostly-dark DigiCam framing; the flag defaults off for general inputs.

### Run inference

```bash
rm -rf outputs/digicam_simple_admm100   # optional: wipe old run

.venv/bin/python inference.py -cn inference
# -cn inference → src/configs/inference.yaml (fixed ADMM)

.venv/bin/python inference.py -cn inference dataset.limit=10   # quick smoke test
```

Outputs land in `outputs/digicam_simple_admm100/recon/` and `.../lensed/`. Heads up: the first sample per new mask is slow (PSF simulation); I cache PSFs per `mask_label` after that.

### Metrics

Scores use the DigiCam scene ROI (`[80:280, 100:366]`), not the full black-bordered sensor frame. For non-DigiCam folders, pass `use_roi=false`.

```bash
.venv/bin/python calculate_metrics.py \
  pred_dir=outputs/digicam_simple_admm100/recon \
  gt_dir=outputs/digicam_simple_admm100/lensed \
  output_path=outputs/digicam_simple_admm100/metrics.json
```

## Training (LeADMM-20)

This is the command I used for LeADMM training with Comet logging:

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

Set `trainer.override=false` when resuming — otherwise I wipe the save dir on a new run. Inference with a checkpoint:

```bash
.venv/bin/python inference.py -cn inference_leadmm20 \
  checkpoint=outputs/train_leadmm20/checkpoint-epochN.pth \
  output_dir=outputs/digicam_leadmm20
```

All configs are in `src/configs/`. I log to Comet via `train_comet.yaml` (API key in `private_tokens.py` or env).

## Bonus: FlatNet-lite

`flatnet_lite` is a non-ADMM baseline for the Trainable Inversion bonus: learned Wiener/Fourier inversion followed by a DRUNet enhancer.

```bash
.venv/bin/python train.py -cn train_comet \
  model=flatnet_lite \
  datasets.train.limit=20 \
  datasets.val.limit=5 \
  trainer.n_epochs=1 \
  optimizer.lr=1e-4 \
  trainer.save_dir=outputs/train_flatnet_lite_debug \
  writer.run_name=flatnet_lite_debug

.venv/bin/python inference.py -cn inference_flatnet \
  checkpoint=outputs/train_flatnet_lite_debug/flatnet_lite_debug/model_best.pth \
  output_dir=outputs/final_flatnet_lite_debug
```
