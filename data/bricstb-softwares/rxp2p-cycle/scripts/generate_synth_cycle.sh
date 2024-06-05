#!/bin/sh
#SBATCH --nodes=1            	# Number of nodes
#SBATCH --ntasks-per-node=1  	# Number of tasks/node
#SBATCH --partition=gpu      	# The partion name: gpu or cpu
#SBATCH --job-name=gen_cycle_santacasa 	# job name
#SBATCH --exclusive         	# Reserve this node only for you
#SBATCH --account=otto.tavares	# account name
#SBATCH --output=output_jobs_train_cycle/Result-%x.%j.out
#SBATCH --error=output_jobs_train_cycle/Result-%x.%j.err
# Runtime of this job

python generate_synth.py --checkpoints_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/checkpoints/  --results_dir /home/otto.tavares/public/iltbi/rxpixp2pixcycle/versions/v1/user.otavares.cycle.shenzhenSantaCasa.v1_BtoA.r1/1aPRODUCAO/  --dataroot /home/brics/public/brics_data/Shenzhen --dataset_download_dir /home/otto.tavares/public/iltbi/train/images --download_imgs --custom_images_path /home/brics/public/brics_data/Shenzhen/raw/images --model test2cycle --direction BtoA --netG resnet_9blocks  --dataset_mode cyclesantacasaskfold2generator --token b16fe0fc92088c4840a98160f3848839e68b1148 --n_folds 10 --preprocess resize_and_scale_width --no_dropout --gpu_ids 0 --name genkl --isTB --gen_test #--gen_train_dataset