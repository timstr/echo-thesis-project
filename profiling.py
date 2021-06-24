import datetime

s_print_results = True

s_profiling_stack = []


class Timer:
    def __init__(self, description):
        assert isinstance(description, str)
        if s_print_results:
            print(f"{'|   ' * len(s_profiling_stack)}{description} started")
        self.description = description
        self.start_time = datetime.datetime.now()
        s_profiling_stack.append(id(self))

    def done(self):
        assert s_profiling_stack[-1] == id(self)
        s_profiling_stack.pop()
        now = datetime.datetime.now()
        duration = now - self.start_time
        seconds = float(duration.seconds) + (duration.microseconds / 1_000_000.0)
        if s_print_results:
            print(
                f"{'|   ' * len(s_profiling_stack)}{self.description} done, {seconds} seconds"
            )
