import warnings

import hydra
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def get_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = get_device(config.get("device", "auto"))
    print(f"using device: {device}")


if __name__ == "__main__":
    main()