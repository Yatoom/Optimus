import signal


class Timeout:
    """Timeout class using ALARM signal."""

    class Timeout(BaseException):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        if self.sec is not None:
            signal.signal(signal.SIGALRM, self.raise_timeout)
            signal.alarm(self.sec)

    def __exit__(self, *args):
        if self.sec is not None:
            signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise TimeoutError
