import warnings

import hydra

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="speed")
def main(config):
    raise NotImplementedError


if __name__ == "__main__":
    main()
