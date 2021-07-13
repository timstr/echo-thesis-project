import textwrap
import os
from config_constants import output_format_depthmap, output_format_heatmap

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(format, implicit, variance, summary_statistics):
    desc = f"train_echo4ch_{format}_{'implicit' if implicit else 'dense'}_{'variance' if variance else 'novariance'}_{'summarystatistics' if summary_statistics else 'nosummarystatistics'}"
    contents = f"""\
    #!/bin/bash

    #PBS -l walltime=36:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
    #PBS -N {desc}
    #PBS -A st-rhodin-1-gpu
    #PBS -m abe
    #PBS -M timstr@cs.ubc.ca
    #PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_OUTPUT.txt
    #PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_ERROR.txt

    source activate /home/timstr/echo_env
    cd /home/timstr/echo

    export ECHO4CH_DATASET=/project/st-rhodin-1/users/timstr/echo4ch
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models

    echo "Starting training..."
    python3 echolearn.py \\
        --experiment={desc} \\
        --dataset=echo4ch \\
        --batchsize=32 \\
        --receivercount=8 \\
        --receiverarrangement=grid \\
        --emitterarrangement=mono \\
        --emittersignal=sweep \\
        --allowocclusions \\
        --nodisplay \\
        {'--implicitfunction' if implicit else ''} \\
        --importancesampling \\
        {'--predictvariance' if variance else ''} \\
        --resolution=64 \\
        --nninput=spectrogram \\
        --nnoutput={format} \\
        {'--summarystatistics' if summary_statistics else ''} \\
        --plotinterval=256 \\
        --validationinterval=1024
    
    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for format in [output_format_depthmap, output_format_heatmap]:
    for implicit in [False, True]:
        for variance in [False, True]:
            for summary_statistics in [False, True]:
                make_script(format, implicit, variance, summary_statistics)
