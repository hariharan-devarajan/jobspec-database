#!/bin/sh
#SBATCH --job-name CEML_sobely_sobelx_original_resnet
#SBATCH --nodes=1
#SBATCH -c 3
#SBATCH -t 72:00:00
#SBATCH -o "CEML_sobely_sobelx_original_resnet.out"
#SBATCH --chdir=/code/CEML/v2

DATA_DIR="../images/dogs_cats_parent/dogs_cats-sobel_x-sobel_y-composit"
CYCLES=20

echo "queue is `echo $SLURM_JOB_PARTITION`"
echo "running on `echo $SLURM_JOB_NODELIST`"
echo "work directory is `echo $SLURM_SUBMIT_DIR`"

echo $(date)
echo "CREATE COMPOSITE"
srun python3 process_images.py -i "../images/dogs_cats_parent/dogs_cats" --sobel_x --sobel_y --original

echo $(date)
echo "RESNET18 MODEL"
srun python3 create_model.py --input $DATA_DIR --model "resnet18" --cycles $CYCLES

echo $(date)
echo "RESNET34 MODEL"
srun python3 create_model.py --input $DATA_DIR --model "resnet34" --cycles $CYCLES

echo $(date)
echo "RESNET50 MODEL"
srun python3 create_model.py --input $DATA_DIR --model "resnet50" --cycles $CYCLES

echo $(date)
echo "RESNET101 MODEL"
srun python3 create_model.py --input $DATA_DIR --model "resnet101" --cycles $CYCLES

echo $(date)
echo "RESNET152 MODEL"
srun python3 create_model.py --input $DATA_DIR --model "resnet152" --cycles $CYCLES
