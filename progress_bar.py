import sys

def progress_bar(current, total):
    i = current + 1
    bar_fill = '=' * (i * 50 // total)
    end_of_line = "\n" if (i == total) else ""
    sys.stdout.write("\r[%-50s] %d/%d%s" % (bar_fill, i, total, end_of_line))
    sys.stdout.flush()