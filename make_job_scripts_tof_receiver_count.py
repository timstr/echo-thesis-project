import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(tof_crop_size, x_count, y_count, z_count):
    desc = f"tof_c{tof_crop_size}_x{x_count}_y{y_count}_z{z_count}"
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

    export WAVESIM_DATASET_TRAIN=/project/st-rhodin-1/users/timstr/dataset_random_train.h5
    export WAVESIM_DATASET_VALIDATION=/project/st-rhodin-1/users/timstr/dataset_random_val_small.h5
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models
    export MPLCONFIGDIR=/scratch/st-rhodin-1/users/timstr/matplotlib_junk

    echo "Starting training..."
    python3 train_time_of_flight_net.py \\
        --experiment={desc} \\
        --batchsize=128 \\
        --iterations=1000000 \\
        --tofcropsize={tof_crop_size} \\
        --samplesperexample=256 \\
        --plotinterval=1024 \\
        --validationinterval=4096 \\
        --receivercountx={x_count} \\
        --receivercounty={y_count} \\
        --receivercountz={z_count}

    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for tof_crop_size in [64, 128]:
    for x_count in [2, 4]:
        for y_count in [n for n in [2, 4] if n >= x_count]:
            for z_count in [1, 2, 4]:
                make_script(
                    tof_crop_size=tof_crop_size,
                    x_count=x_count,
                    y_count=y_count,
                    z_count=z_count,
                )
