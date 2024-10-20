#!/bin/bash
#SBATCH --job-name=train_yiming_amass
#SBATCH -D .
#SBATCH --output=slurm/%x_%j_O.log
#SBATCH --error=slurm/%x_%j_E.log
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:1            # number of GPUs per node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --time=1-00:00:00           # maximum execution time (HH:MM:SS)


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/kostas-graid/sw/envs/fengqiao/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

PATH="/mnt/kostas-graid/sw/envs/fengqiao/local/cuda-12.1/bin:$PATH"
LD_LIBRARY_PATH="/mnt/kostas-graid/sw/envs/fengqiao/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"


conda activate isaac
bash train_yiming_amass.sh