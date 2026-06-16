# Lensless Computational Imaging

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_digicam_masks.py   # mask_0.npy … mask_99.npy into data/digicam_masks/
```

## Inference

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

```bash
.venv/bin/python calculate_metrics.py \
  pred_dir=outputs/digicam_simple_admm100/recon \
  gt_dir=outputs/digicam_simple_admm100/lensed \
  output_path=outputs/digicam_simple_admm100/metrics.json
```

## Training

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

```bash
.venv/bin/python train.py -cn train_comet \
  model=modular_leadmm5_prepost_drunet8m \
  trainer.n_epochs=10 \
  trainer.save_dir=outputs/train_leadmm5_drunet8m_prepost_ep10 \
  writer.run_name=leadmm5_drunet8m_prepost_ep10
```

## FlatNet-Lite

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

## Custom Data

```bash
.venv/bin/python inference.py -cn inference_custom \
  dataset.root=/path/to/data \
  output_dir=outputs/custom_run
```
