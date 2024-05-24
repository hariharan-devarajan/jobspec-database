#!/bin/bash -l

#SBATCH -p datasci


#SBATCH --job-name=celeba_train
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
cd ..

# scd01


#python train_mlp_major_.py --dataset celeba --n_classes 2 --iters 200 --hidden-nodes 20 --lr 0.01 --save --seed 2018 \
#--round 100 --target celeba_100_mlp_logistic_20.pkl

python train_lenet_.py --dataset celeba --seed 2018 --n_classes 2 \
--save --target celeba_resnet50_10.pkl --round 10
#
#python train_lenet_.py --dataset celeba --seed 2018 --n_classes 2 \
#--save --target celeba_lenet_100.pkl --round 100

#python train_binarynn.py --dataset celeba --n_classes 2 --hidden-nodes 20 --seed 2018 \
#--target celeba_mlpbnn_approx --round 100

#python train_scd.py --nrows 0.1 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 32 --interval 20 --n-jobs 4 --num-gpus 4 --save --n_classes 2 --iters 1 \
#--target celeba_scd01mlp_1_br02_nr010_ni1000_i1_h20.pkl --dataset celeba --version mlp --seed 2019 --width 1000 \
#--metrics balanced
#
#
#python train_scd.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 20 --b-ratio 0.2 \
#--updated-features 64 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_scd01_1_br02_nr075_ni1000_i1.pkl --dataset mnist  --seed 2018
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.17 --num-iters 50 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_hinge_10_normal.pkl --dataset celeba  --seed 2018 --c 1
#

#for seed in 0 1 2 3 4 5 6 7 8 9 10 11
#do

#python train_mlphinge.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 1000 --b-ratio 0.2 \
#--updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_mlp_hinge_reluact_c1_1000_$seed.pkl --dataset celeba  --seed $seed --c 1 --act relu

#python train_mlplogistic.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_mlp_logistic_reluact_1000_$seed.pkl --dataset celeba  --seed $seed --act relu

#python train_mlplogistic.py --nrows 0.25 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_mlp_logistic_reluact_1000_${seed}.pkl --dataset gtsrb_binary  --seed $seed --act relu

#python train_mlphinge.py --nrows 0.25 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.2 --hidden-nodes 20 \
#--num-iters 1000 --b-ratio 0.2 \
#--updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_mlp_hinge_reluact_c1_1000_${seed}.pkl --dataset gtsrb_binary  --seed $seed --c 1 --act relu
#
#python train_mlp01.py --nrows 0.25 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_mlp_01_01act_1000_${seed}.pkl --dataset gtsrb_binary  --seed $seed  --act sign
#
#python train_mlp01.py --nrows 0.25 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 256 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_mlp_01_reluact_1000_${seed}.pkl --dataset gtsrb_binary  --seed $seed  --act relu
#
#python train_scd.py --nrows 0.25 --nfeatures 1 --w-inc 0.05 --num-iters 1000 --b-ratio 0.2 --updated-features 256 \
#--round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --target gtsrb_binary_scd_nr025_i10_1000_${seed}.pkl \
#--dataset gtsrb_binary --seed $seed

#
#python train_mlp01.py --nrows 0.75 --nfeatures 1 --w-inc1 0.17 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target gtsrb_binary_mlp_01_01act_1000_$seed.pkl --dataset gtsrb_binary  --seed $seed  --act sign

#done
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.2 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w02_1000_2018.pkl --dataset celeba \
# --seed 2018 --c 1

#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.1 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w01_1000_2018.pkl --dataset celeba \
# --seed 2018 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.5 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w05_1000_2018.pkl --dataset celeba \
# --seed 2018 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w005_1000_2018.pkl --dataset celeba \
# --seed 2018 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.2 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w02_1000_2019.pkl --dataset celeba \
# --seed 2019 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.1 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w01_1000_2019.pkl --dataset celeba \
# --seed 2019 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.5 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w05_1000_2019.pkl --dataset celeba \
# --seed 2019 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w005_1000_2019.pkl --dataset celeba \
# --seed 2019 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.2 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w02_1000_2020.pkl --dataset celeba \
# --seed 2020 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.1 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w01_1000_2020.pkl --dataset celeba \
# --seed 2020 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.5 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w05_1000_2020.pkl --dataset celeba \
# --seed 2020 --c 1
#
#python train_hinge.py --nrows 0.75 --nfeatures 1 --w-inc 0.05 --num-iters 1000 --b-ratio 0.2 --updated-features 128 \
#--round 1 --interval 10 --gpu 0  --save --n_classes 2 --iters 1 --target celeba_hinge_c1_w005_1000_2020.pkl --dataset celeba \
# --seed 2020 --c 1
#

#python train_mlpl.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_mlp_l_reluact_1000_$seed_20.pkl --dataset celeba  --seed $seed --act relu --updated-nodes 20
#
#python train_mlpl.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_mlp_l_signact_1000_$seed_20.pkl --dataset celeba  --seed $seed --act sign --updated-nodes 20
#
#
#
#python test.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 10 \
#--b-ratio 0.2 --updated-features 2 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 \
#--target celeba_mlp_l_reluact_1000_$seed.pkl --dataset celeba  --seed $seed --act relu
#
#python test2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 30 \
#--b-ratio 0.2 --updated-features 2 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target circle_1_nr075_${version}_sign_01loss_30_0_w1_h1_005005.pkl --dataset circle  --seed 0  \
#--version $version --act sign --width 1
#
#python train_mlp_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.2 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target celeba_1_nr075_${version}_sign_01loss_1000_2018_w1_h1_0202.pkl --dataset celeba  --seed 2018  \
#--version $version --act sign --width 1
#
#
#python train_mlp_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.2 --w-inc2 0.2 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target celeba_1_nr075_${version}_relu_01loss_1000_2018_w1_h1_0202.pkl --dataset celeba  --seed 2018  \
#--version $version --act relu --width 1
#
#python train_mlp_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.1 --w-inc2 0.1 --hidden-nodes 20 --num-iters 100 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target celeba_1_nr075_${version}_relu_01loss_100_2018_w1_h1_0101.pkl --dataset celeba  --seed 2018  \
#--version $version --act relu --width 10
#
#python train_mlp_logistic_ml_2.py --nrows 0.75 --nfeatures 1 --w-inc1 0.05 --w-inc2 0.05 --hidden-nodes 20 --num-iters 1000 \
#--b-ratio 0.2 --updated-features 128 --round 1 --interval 20 --gpu 0  --save --n_classes 2 --iters 1 --updated-nodes 1 \
#--target celeba_1_nr075_${version}_relu_logistic_1000_2018_w1_h1_005005.pkl --dataset celeba  --seed 2018  \
#--version $version --act relu --width 1 --status sigmoid