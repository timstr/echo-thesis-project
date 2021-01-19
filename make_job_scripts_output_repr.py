import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    
def make_script(format, implicit, variance):
    desc = f"train_output_repr_4_{format}_{'implicit' if implicit else 'dense'}_{'variance' if variance else 'novariance'}"
    contents = f"""\
    #!/bin/bash

    #PBS -l walltime=12:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
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
        --batchsize=64 \\
        --receivercount=8 \\
        --receiverarrangement=grid \\
        --emitterarrangement=mono \\
        --emittersignal=sweep \\
        --allowocclusions \\
        --nodisplay \\
        {'--implicitfunction' if implicit else ''} \\
        --importancesampling \\
        {'--predictvariance' if variance else ''} \\
        --resolution=256 \\
        --nninput=spectrogram \\
        --nnoutput={format} \\
        --summarystatistics \\
        --plotinterval=64 \\
        --validationinterval=1024
    
    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline='\n') as f:
        f.write(contents)

for format in ["depthmap", "heatmap", "sdf"]:
    for implicit in [False, True]:
        for variance in [False, True]:
            make_script(format, implicit, variance)