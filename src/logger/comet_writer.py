from src.utils.comet_credentials import get_comet_credentials


class CometWriter:

    def __init__(
        self,
        logger,
        project_config,
        project_name="lensless-computational-imaging",
        workspace=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        self.logger = logger
        self.step = 0
        self.mode = "train"
        self.exp = None

        try:
            import comet_ml
        except ImportError as exc:
            raise ImportError(
                "comet_ml is required for CometWriter. Install with: pip install comet_ml"
            ) from exc

        creds = get_comet_credentials(project_name=project_name, workspace=workspace)
        api_key = creds.api_key

        if mode == "online" and not api_key:
            raise ValueError(
                "Comet API key is required for writer.mode=online. "
                "Set COMET_API_KEY in private_tokens.py, export COMET_API_KEY, "
                "use writer=noop, or set writer.mode=offline."
            )

        resolved_project = creds.project_name or project_name
        resolved_workspace = creds.workspace or workspace

        if api_key:
            comet_ml.login(api_key=api_key)

        writer_cfg = project_config.get("writer", {}) if project_config else {}
        trainer_cfg = project_config.get("trainer", {}) if project_config else {}
        run_id = writer_cfg.get("run_id")
        resume = trainer_cfg.get("resume_from") is not None

        exp_kwargs = {
            "project_name": resolved_project,
            "workspace": resolved_workspace,
            "log_code": False,
            "log_graph": False,
            "auto_metric_logging": False,
            "auto_param_logging": False,
        }
        if api_key:
            exp_kwargs["api_key"] = api_key
        if run_id:
            exp_kwargs["experiment_key"] = run_id

        if resume and run_id:
            exp_class = (
                comet_ml.ExistingOfflineExperiment
                if mode == "offline"
                else comet_ml.ExistingExperiment
            )
            self.exp = exp_class(experiment_key=run_id, api_key=api_key)
        elif mode == "offline":
            self.exp = comet_ml.OfflineExperiment(**exp_kwargs)
        else:
            self.exp = comet_ml.Experiment(**exp_kwargs)

        if run_name:
            self.exp.set_name(run_name)
        self.exp.log_parameters(parameters=project_config)

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def _metric_name(self, name):
        return f"{name}_{self.mode}"

    def add_scalar(self, name, value):
        if self.exp is None:
            return
        self.exp.log_metric(self._metric_name(name), value, step=self.step)

    def add_image(self, name, image):
        if self.exp is None:
            return
        image_data = self._image_to_numpy(image)
        self.exp.log_image(
            image_data=image_data,
            name=self._metric_name(name),
            step=self.step,
        )

    def add_checkpoint(self, checkpoint_path, save_dir):
        if self.exp is None:
            return
        self.exp.log_model(
            name="checkpoints",
            file_or_folder=str(checkpoint_path),
            overwrite=True,
        )

    def close(self):
        if self.exp is not None:
            self.exp.end()

    @staticmethod
    def _image_to_numpy(image):
        import numpy as np
        import torch

        if torch.is_tensor(image):
            x = image.detach().cpu().clamp(0, 1)
            if x.ndim == 4:
                x = x[0]
            if x.ndim == 3 and x.shape[0] in (1, 3):
                if x.shape[0] == 1:
                    return x[0].numpy()
                return x.permute(1, 2, 0).numpy()
            raise ValueError(f"unsupported image tensor shape: {tuple(x.shape)}")
        return np.asarray(image)
