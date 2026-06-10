from abc import abstractmethod


class BaseMetric:

    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        raise NotImplementedError()
