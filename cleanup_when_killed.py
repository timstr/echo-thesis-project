import signal
import sys


def sigterm_handler(signo, stack_frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, sigterm_handler)
