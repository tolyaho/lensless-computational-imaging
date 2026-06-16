class NoOpWriter:

    def __init__(self, logger, project_config, **kwargs):
        self.step = 0
        self.mode = "train"

    def set_step(self, step, mode="train"):
        self.step = step
        self.mode = mode

    def add_scalar(self, scalar_name, scalar):
        pass

    def add_image(self, image_name, image):
        pass

    def add_checkpoint(self, checkpoint_path, save_dir):
        pass

    def close(self):
        pass
