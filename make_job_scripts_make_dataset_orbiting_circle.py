import textwrap
import os

out_folder = "job_scripts"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)


def make_script(numworkers, workerindex):
    desc = f"make_dataset_orbiting_circle_{workerindex+1}_of_{numworkers}"
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

    export OUTPUT_PATH=/scratch/st-rhodin-1/users/timstr/echo/output_dataset_orbiting_circle

    echo "Starting dataset creation..."
    python3 make_dataset.py \\
        --offset=0 \\
        --end=10000 \\
        --numworkers={numworkers} \\
        --workerindex={workerindex} \\
        --mode=orbiting-circle
    
    echo "Dataset creation completed."
    """
    contents = textwrap.dedent(contents)
    script_name = f"{desc}.sh"
    with open(os.path.join(out_folder, script_name), "w", newline="\n") as f:
        f.write(contents)


nworkers = 4
for i in range(nworkers):
    make_script(nworkers, i)
