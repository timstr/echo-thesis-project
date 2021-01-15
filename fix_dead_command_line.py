# Must be imported before pytorch

# The following fixes a zombie command line
# Credit goes to https://stackoverflow.com/a/44822794/5023438
import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"