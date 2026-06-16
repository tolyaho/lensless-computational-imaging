from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.model.flatnet import FlatNetLite, LearnedWienerInversion


def test_flatnet_lite_preserves_shape_and_gradients():
    model = FlatNetLite(
        inversion=LearnedWienerInversion(channels=3),
        clamp_output=False,
    )
    lensless = torch.rand(1, 3, 32, 33, requires_grad=True)
    mask = torch.rand(1, 3, 32, 33)

    out = model(lensless=lensless, mask=mask)

    assert set(out) == {"recon", "flatnet_intermediate"}
    assert out["recon"].shape == lensless.shape
    assert out["flatnet_intermediate"].shape == lensless.shape

    loss = out["recon"].mean() + out["flatnet_intermediate"].mean()
    loss.backward()

    assert lensless.grad is not None
    assert model.inversion.raw_lambda.grad is not None
    assert model.inversion.scale.grad is not None
    assert model.inversion.bias.grad is not None


def test_flatnet_lite_hydra_model_config_instantiates():
    config_dir = str(Path(__file__).resolve().parents[1] / "src" / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="train", overrides=["model=flatnet_lite"])

    model = instantiate(config.model)

    assert isinstance(model, FlatNetLite)
