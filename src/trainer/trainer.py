from pathlib import Path

import torch
import torch.nn.functional as F

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.image_io import save_image


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        if self.metrics is not None:
            metric_funcs = (
                self.metrics["train"] if self.is_train else self.metrics["inference"]
            )
        else:
            metric_funcs = []

        if self.is_train:
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(**batch)

        if not isinstance(outputs, dict):
            outputs = {"recon": outputs}

        batch.update(outputs)

        all_losses = self.criterion(**batch)

        if torch.is_tensor(all_losses):
            all_losses = {"loss": all_losses}

        batch.update(all_losses)

        if not torch.isfinite(batch["loss"]):
            raise RuntimeError(f"Non-finite loss: {batch['loss'].item()}")

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            return

        if batch_idx != 0:
            return

        self._save_sample_images(batch, mode)
        self._log_admm_params(mode)

    def _sample_dir(self, mode: str) -> Path:
        return self.checkpoint_dir / "samples" / mode

    def _save_sample_images(self, batch, mode: str):
        if "recon" not in batch:
            return

        out_dir = self._sample_dir(mode)
        out_dir.mkdir(parents=True, exist_ok=True)

        image_ids = batch.get("image_id", [])
        if not isinstance(image_ids, list):
            image_ids = [str(image_ids)]

        n_save = min(2, len(image_ids), batch["recon"].shape[0])
        fields = ("lensless", "recon", "lensed")

        for i in range(n_save):
            image_id = image_ids[i]
            for field in fields:
                if field not in batch:
                    continue
                tensor = batch[field][i].detach().clamp(0, 1).cpu()
                path = out_dir / f"{image_id}_{field}.png"
                save_image(tensor, path)

                if self.writer is not None and hasattr(self.writer, "add_image"):
                    self.writer.add_image(f"{mode}/{image_id}_{field}", tensor)

    def _log_admm_params(self, mode: str):
        if not hasattr(self.model, "raw_mu") or not hasattr(self.model, "raw_tau"):
            return

        eps = getattr(self.model, "eps", 0.0)
        mu = F.softplus(self.model.raw_mu).detach() + eps
        tau = F.softplus(self.model.raw_tau).detach() + eps

        stats = {
            "admm/mu_mean": mu.mean().item(),
            "admm/mu_min": mu.min().item(),
            "admm/mu_max": mu.max().item(),
            "admm/tau_mean": tau.mean().item(),
            "admm/tau_min": tau.min().item(),
            "admm/tau_max": tau.max().item(),
        }

        if self.writer is not None:
            for name, value in stats.items():
                if hasattr(self.writer, "add_scalar"):
                    self.writer.add_scalar(name, value)
