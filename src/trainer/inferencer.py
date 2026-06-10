import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]], writer=None
            )
        else:
            self.evaluation_metrics = None
        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self.model(**batch)
        batch.update(outputs)
        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))
        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size
        for i in range(batch_size):
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)
            output_id = current_id + i
            output = {"pred_label": pred_label, "label": label}
            if self.save_path is not None:
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")
        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
        return self.evaluation_metrics.result()
