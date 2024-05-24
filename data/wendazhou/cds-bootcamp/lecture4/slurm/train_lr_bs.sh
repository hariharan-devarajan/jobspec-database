#! /bin/bash
#SBATCH --time 1:00:00
#SBATCH -c 4
#SBATCH --mem 6GB
#SBATCH --gres=gpu:1


# Set this to the path of the singularity image you wish to use if different from default
IMAGE=${IMAGE:-/scratch/wz2247/singularity/images/pytorch_21.06-py3.sif}
# Set this to the directory containing your package overlays, by default we are using the directory
# from lecture 2
OVERLAY_DIR=${OVERLAY_DIR:-../lecture2}

# We make use of the multirun functionality of hydra to run the training script
# with batch size from 64 to 4096 in increments of powers of two.

# Note that compared to the `start_singularity` script in lecture2,
# I am only using the overlays containing the packages here (and also not loading the data overlays),
# to keep my dependencies minimal.


# Try setting the parameter scale_lr_by_bs to False to see how batch size affects results
# when learning rate is not scaled correctly.
singularity exec --no-home -B $HOME/.ssh -B /scratch -B $PWD --nv \
    --cleanenv \
    --overlay $OVERLAY_DIR/overlay-base.ext3:ro \
    --overlay $OVERLAY_DIR/overlay-packages.ext3:ro \
    $IMAGE /bin/bash << 'EOF'
source ~/.bashrc
conda activate /ext3/conda/bootcamp
python -um bootcamp.train_lr --multirun batch_size=128,256,512,1024,2048,4096 scale_lr_by_bs=True
EOF

