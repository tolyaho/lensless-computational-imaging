import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config: DictConfig):
    set_random_seed(config.trainer.seed)

    logger = setup_saving_and_logging(config)
    project_config = OmegaConf.to_container(config, resolve=True)

    writer = instantiate(config.writer, logger, project_config)

    device = get_device(config.trainer.device)
    logger.info(f"Using device: {device}")

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(model)
    logger.info(f"Trainable parameters: {count_trainable_parameters(model):,}")

    loss_function = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    lr_scheduler = None
    if config.get("lr_scheduler") is not None:
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
