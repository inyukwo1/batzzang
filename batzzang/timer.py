import time


class Timer:
    start_time = None
    pause_time = None
    elapsed_time = 0.0

    @classmethod
    def show_elapsed_time(cls):
        assert Timer.start_time is None
        assert Timer.pause_time is None
        print(f"Elapsed time: {Timer.elapsed_time}")

    def __del__(self):
        if Timer.elapsed_time != 0.0:
            Timer.show_elapsed_time()


class Pause:
    def __enter__(self):
        if Timer.start_time is not None:
            Timer.pause_time = time.time()

    def __exit__(self, type, value, traceback):
        if Timer.start_time is not None:
            Timer.start_time += (time.time() - Timer.pause_time)
            Timer.pause_time = None


def measure_time(func):
    def wrapper(*args, **kwargs):
        if Timer.start_time is None:
            # Set timer
            Timer.start_time = time.time()
            # Compute
            result = func(*args, **kwargs)
            # Record time
            Timer.elapsed_time += (time.time() - Timer.start_time)
            Timer.start_time = None
            return result
        else:
            return func(*args, **kwargs)
    return wrapper