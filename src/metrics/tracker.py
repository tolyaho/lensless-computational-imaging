class MetricTracker:

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._keys = list(keys)
        self._totals = {key: 0.0 for key in self._keys}
        self._counts = {key: 0 for key in self._keys}

    def reset(self):
        for key in self._keys:
            self._totals[key] = 0.0
            self._counts[key] = 0

    def update(self, key, value, n=1):
        self._totals[key] += float(value) * n
        self._counts[key] += n

    def avg(self, key):
        if self._counts[key] == 0:
            return 0.0
        return self._totals[key] / self._counts[key]

    def result(self):
        return {key: self.avg(key) for key in self._keys}

    def keys(self):
        return self._keys
