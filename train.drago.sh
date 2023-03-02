#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH -n 8

module load GCC/10.2.0
module load CUDA/11.1.1

# This seems to be required since the code in .bashrc does not work in submitted jobs
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre/home/iaa/aluque/mambaforge/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/home/iaa/aluque/mambaforge/etc/profile.d/conda.sh" ]; then
        . "/lustre/home/iaa/aluque/mambaforge/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/home/iaa/aluque/mambaforge/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/lustre/home/iaa/aluque/mambaforge/etc/profile.d/mamba.sh" ]; then
    . "/lustre/home/iaa/aluque/mambaforge/etc/profile.d/mamba.sh"
fi
# <<< conda initialize <<<


mamba activate
python train.py "$@"

