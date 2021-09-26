import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(use_convolutions, use_fourier_transform, hidden_features, kernel_size):
    desc_cnn_fc = "cnn" if use_convolutions else "fc"
    desc_kernel = f"_k{kernel_size}" if use_convolutions else ""
    desc_fdtd = "_fd" if use_fourier_transform else "_td"
    desc_feat = f"_h{hidden_features}"
    desc = desc_cnn_fc + desc_kernel + desc_fdtd + desc_feat
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

    export WAVESIM_DATASET_TRAIN=/project/st-rhodin-1/users/timstr/dataset_7.5mm_random_train.h5
    export WAVESIM_DATASET_VALIDATION=/project/st-rhodin-1/users/timstr/dataset_7.5mm_random_val.h5
    export TRAINING_LOG_PATH=/scratch/st-rhodin-1/users/timstr/echo/logs
    export TRAINING_MODEL_PATH=/scratch/st-rhodin-1/users/timstr/echo/models
    export MPLCONFIGDIR=/scratch/st-rhodin-1/users/timstr/matplotlib_junk

    echo "Starting training..."
    python3 train_model.py \\
        --experiment={desc} \\
        --batchsize=32 \\
        --iterations=1000000 \\
        --tofcropsize=256 \\
        --samplesperexample=256 \\
        --plotinterval=1024 \\
        --validationinterval=4096 \\
        --receivercountx=1 \\
        --receivercounty=2 \\
        --receivercountz=2 \\
        --chirpf0=18000.0 \\
        --chirpf1=22000.0 \\
        --chirplen=0.001 \\
        --validationdownsampling=4 \\
        --use_convolutions={use_convolutions} \\
        --use_fourier_transform={use_fourier_transform} \\
        --hidden_features={hidden_features} \\
        --kernel_size={kernel_size}

    echo "Training completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


for hidden_features in [64, 128, 256]:
    for use_convolutions, kernel_size in [
        (False, 0),
        (True, 5),
        (True, 15),
        (True, 31),
    ]:
        for use_fourier_transform in [False, True]:
            make_script(
                hidden_features=hidden_features,
                kernel_size=kernel_size,
                use_convolutions=use_convolutions,
                use_fourier_transform=use_fourier_transform,
            )
