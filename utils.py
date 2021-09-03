import sys


def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1) == 0) and (n != 0)


print_width = 50


def fancy_horizontal_rule():
    sys.stdout.write("|" + "-" * (print_width - 2) + "|\n")
    sys.stdout.flush()


def fancy_print(contents=None):
    if contents is None:
        contents = ""
    n = len(contents)
    l = " " * ((print_width - 2 - n) // 2)
    m = max(0, print_width - 2 - len(l) - len(contents))
    r = " " * m
    sys.stdout.write("|" + l + contents + r + "|\n")
    sys.stdout.flush()


def progress_bar(current, total):
    i = current + 1
    if (not sys.stdout.isatty()) and (i < total):
        return
    bar_fill = "=" * (i * 50 // total)
    end_of_line = "\n" if (i == total) else ""
    sys.stdout.write("\r[%-50s] %d/%d%s" % (bar_fill, i, total, end_of_line))
    sys.stdout.flush()
