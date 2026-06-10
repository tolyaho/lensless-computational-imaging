from pathlib import Path

import numpy as np
import torch
from PIL import Image


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.array(img)
    x = torch.from_numpy(arr).float() / 255.0

    return x.permute(2, 0, 1).contiguous()


def load_image(path: str | Path) -> torch.Tensor:
    path = Path(path)

    with Image.open(path) as img:
        return image_to_tensor(img)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = tensor.detach().cpu()

    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError("expected single image or batch of size 1")
        x = x[0]

    if x.ndim != 3:
        raise ValueError("expected tensor with shape [C, H, W]")

    x = x.clamp(0, 1)

    if x.shape[0] == 1:
        arr = x[0].mul(255).round().byte().numpy()
        img = Image.fromarray(arr, mode="L")
    elif x.shape[0] == 3:
        arr = x.permute(1, 2, 0).mul(255).round().byte().numpy()
        img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError("expected 1 or 3 channels")

    img.save(path)


def load_mask(path: str | Path) -> torch.Tensor:
    path = Path(path)

    arr = np.load(path)
    is_int = np.issubdtype(arr.dtype, np.integer)

    x = torch.from_numpy(np.array(arr)).float()

    if is_int:
        x = x / 255.0

    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] in (1, 3):
            pass
        elif x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"cannot infer mask layout for shape {tuple(x.shape)}")
    else:
        raise ValueError(f"expected 2d or 3d mask, got shape {tuple(x.shape)}")

    return x.contiguous()
