import textwrap
import os
from config_constants import input_format_audioraw, input_format_gcc

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(tof_crop_size, z_count, xy_count):
    desc = f"tof_c{tof_crop_size}_z{z_count}_xy{xy_count}"
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

    export WAVESIM_DATASET=/project/st-rhodin-1/users/timstr/dataset_random.h5
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models
    export MPLCONFIGDIR=/scratch/st-rhodin-1/users/timstr/matplotlib_junk

    echo "Starting training..."
    python3 train_time_of_flight_net.py \\
        --experiment={desc} \\
        --batchsize=32 \\
        --iterations=1000000 \\
        --tofcropsize={tof_crop_size} \\
        --samplesperexample=128 \\
        --nodisplay \\
        --plotinterval=4096 \\
        --validationinterval=4096 \\
        --receivercountx={xy_count} \\
        --receivercounty={xy_count} \\
        --receivercountz={z_count}

    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for tof_crop_size in [64, 128, 256]:
    for z_count in [1, 2, 4]:
        for xy_count in [2, 4]:
            make_script(tof_crop_size, z_count, xy_count)