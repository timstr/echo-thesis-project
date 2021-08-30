from math import ceil, log10
import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(mode, worker_index, num_workers, count, append):
    worker_index_str = str(worker_index + 1).zfill(ceil(log10(num_workers)))
    desc = f"make_dataset_{mode}_{worker_index_str}_of_{num_workers}"
    contents = f"""\
    #!/bin/bash

    #PBS -l walltime=36:00:00,select=1:ncpus=4:ngpus=1:mem=16gb
    #PBS -N {desc}
    #PBS -A st-rhodin-1-gpu
    #PBS -m abe
    #PBS -M timstr@cs.ubc.ca
    #PBS -o /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_OUTPUT.txt
    #PBS -e /scratch/st-rhodin-1/users/timstr/echo/logs/{desc}_ERROR.txt

    module load cuda openmpi

    source activate /home/timstr/echo_env
    cd /home/timstr/echo

    export ECHO4CH_OBSTACLES=/project/st-rhodin-1/users/timstr/echo4ch_obstacles.h5
    export DATASET_OUTPUT=/scratch/st-rhodin-1/users/timstr/echo/dataset_7.5mm_{mode}_{worker_index_str}_of_{num_workers}.h5
    export KWAVE_EXECUTABLE=/home/timstr/k-wave/kspaceFirstOrder-CUDA/kspaceFirstOrder-CUDA
    export KWAVE_TEMP_FOLDER=/scratch/st-rhodin-1/users/timstr/echo/temp/{desc}

    echo "Starting dataset generation..."
    python3 make_dataset_3d.py \\
        --numworkers={num_workers} \\
        --workerindex={worker_index} \\
        --mode={mode} \\
        --count={count} \\
        {'--append' if append else ''}

    echo "Dataset generation completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


append = True

if append:
    print("WARNING: jobs will append to existing datasets")

num_workers = 20
for mode, count in [("echo4ch", 6500)]:
    for worker_index in range(num_workers):
        make_script(
            mode=mode,
            worker_index=worker_index,
            num_workers=num_workers,
            count=count,
            append=append,
        )
