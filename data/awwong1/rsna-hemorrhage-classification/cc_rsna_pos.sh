#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=0-10:00
#SBATCH --mail-type=ALL
#SBATCH --output=%x.out
#SBATCH --account=def-nilanjan
#SBATCH --mail-user=awwong1@ualberta.ca
module load arch/avx512 StdEnv/2018.3
nvidia-smi

source venv/bin/activate

curdir=$PWD
datasetdir=${SLURM_TMPDIR}/rsna/
echo "Moving RSNA Dataset to ${datasetdir}"

# move the kaggle rsna data over to the SLURM tmpdir
mkdir -p $datasetdir
time pv ~/scratch/datasets/rsna-intracranial-hemorrhage-detection.tar | tar xf - -C $datasetdir

for iter in 1 2 3
do
    echo "run ${iter}"
    model="resnet34"
    python3 main.py --rsna-base=${datasetdir}rsna-intracranial-hemorrhage-detection \
	    --batch-size=256 --epochs=5 --test-batch-size=128 \
	    --model=${model} --checkpoint=${model}.${iter}.pos.pth --submission=submission.${model}.${iter}.pos.csv \
	    --tb-log=runs/${model}.${iter}.pos \
	    --apply-pos-weight
done
