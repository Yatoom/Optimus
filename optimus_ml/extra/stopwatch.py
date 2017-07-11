import time


class Stopwatch:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type_, value, tb):
        self.end = time.time()
        self.duration = self.end - self.start

    def elapsed_time(self):
        if self.end is None:
            return time.time() - self.start
        return self.duration
