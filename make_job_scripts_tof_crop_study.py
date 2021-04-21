import textwrap
import os
from config import input_format_audioraw, input_format_gcc

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(format, window_size):
    desc = f"tofc_study_{format}_{window_size}"
    contents = f"""\
    #!/bin/bash

    #PBS -l walltime=24:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
    #PBS -N {desc}
    #PBS -A st-rhodin-1-gpu
    #PBS -m abe
    #PBS -M timstr@cs.ubc.ca
    #PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_OUTPUT.txt
    #PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_ERROR.txt

    source activate /home/timstr/echo_env
    cd /home/timstr/echo

    export WAVESIM_DATASET=/project/st-rhodin-1/users/timstr/wavesim_dataset
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models

    echo "Starting training..."
    python3 echolearn.py \\
        --experiment={desc} \\
        --batchsize=32 \\
        --receivercount=8 \\
        --receiverarrangement=grid \\
        --emitterarrangement=mono \\
        --emittersignal=sweep \\
        --allowocclusions \\
        --nodisplay \\
        --predictvariance \\
        --resolution=256 \\
        --nninput={format} \\
        --nnoutput=sdf \\
        --tofcropping \\
        --tofwindowsize={window_size} \\
        --samplesperexample=32 \\
        --plotinterval=256 \\
        --validationinterval=512
    
    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for format in [input_format_audioraw, input_format_gcc]:
    for window_size in [64, 128, 256]:
        make_script(format, window_size)
