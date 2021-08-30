# Must be imported before pytorch

import sys

if "torch" in sys.modules:
    print(
        "fix_dead_command_line.py was imported after PyTorch. Please import it before PyTorch to allow graceful shutdown after a keyboard interrupt."
    )
    exit(-1)

# The following fixes a zombie command line
# Credit goes to https://stackoverflow.com/a/44822794/5023438
import os

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
