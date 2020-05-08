import sys

def progress_bar(current, total):
    bar_fill = '=' * ((current + 1) * 50 // total)
    sys.stdout.write("\r[%-50s] %d/%d" % (bar_fill, (current + 1), total))
    sys.stdout.flush()