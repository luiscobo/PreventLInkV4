import time
from threading import Timer


class RepeatedTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


def say_hello():
    x = time.time()
    print(f"{x} Hola")


if __name__ == '__main__':
    t = RepeatedTimer(2, say_hello)
    t.start()
    time.sleep(10)
    t.cancel()