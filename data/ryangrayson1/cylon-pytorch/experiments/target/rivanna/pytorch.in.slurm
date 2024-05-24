#!/usr/bin/env bash

#SBATCH --job-name=pytorch-{experiment.batch}-n-{experiment.ngpus}-g-{experiment.card_name}-c-{experiment.concurrency}-{experiment.repeat}
#SBATCH --output=pytorch-{experiment.repeat}-%u-%j.out
#SBATCH --error=pytorch-{experiment.repeat}-%u-%j.err
{slurm.ee}
#SBATCH --nodes={ee.nodes}
#SBATCH --ntasks={ee.ntasks}
#SBATCH --mem={ee.mem}
#SBATCH --gres=gpu:{experiment.card_name}:{experiment.ngpus}
#SBATCH --cpus-per-task={experiment.ncpus}
#SBATCH --time={ee.time}


# xSBATCH --partition=gpu
# xSBATCH --mem=64GB


PROGRESS () {
    echo "# ###########################################"
    echo "# cloudmesh status="$1" progress=$2 pid=$$"
    echo "# ###########################################"
}

PROGRESS "running" 1

echo "# ==================================="
echo "# SLURM info"
echo "# ==================================="

echo USER {os.USER}
echo HOME {os.HOME}
echo cardname {experiment.card_name}
echo ngpus {experiment.ngpus}
echo repeat {experiment.repeat}
echo jobno $SLURM_JOB_ID
echo {slurm.ee}
echo mem {ee.mem}
echo USER $USER
echo {data.sif_dir}

PROGRESS "running" 2

echo "# ==================================="
echo "# PROJECT info"
echo "# ==================================="

USER_PROJECT=/project/bii_dsc_community/$USER
PYTORCH_SIF={data.pytorch_sif}

echo USER_PROJECT $USER_PROJECT
echo PYTORCH_SIF

mkdir -p {data.output}

PROGRESS "running" 3

module purge
module load singularity gcc/11.2.0 openmpi/4.1.4 python/3.11.1

source $USER_PROJECT/ENV/bin/activate

echo "executing multigpu example..."
singularity exec --nv {data.pytorch_sif} python multigpu.py {experiment.total_epochs} {experiment.save_every} --batch_size={experiment.batch}

echo "executing distributed simple example..."
singularity exec --nv {data.pytorch_sif} python distributed_simple.py

echo "executing distributed training example..."
singularity exec --nv {data.pytorch_sif} python distributed_training.py

echo "executing distributed mpi example..."
singularity exec --nv {data.pytorch_sif} mpirun -n 4 python distributed_mpi.py --oversubscribe

echo "==================================================="
seff $SLURM_JOB_ID
