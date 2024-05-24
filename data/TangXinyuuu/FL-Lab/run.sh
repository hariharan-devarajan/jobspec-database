#!/bin/bash
#SBATCH -o /home/tangm_lab/cse12011439/codes/FLlab.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab02      # 作业提交的指定分区队列为titan
#SBATCH -J FLJob       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2          # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制;

python experiments1.py --model=resnet \
	--dataset=cifar10 \
	--alg=fednova \
	--lr=0.01 \
	--batch-size=64 \
	--epochs=10 \
	--n_parties=5 \
	--rho=0.9 \
	--comm_round=50 \
	--partition=noniid-labeldir \
	--beta=0.5\
	--device='cuda:0'\
	--datadir='./data/' \
	--logdir='./logs/' \
	--noise=0\
	--init_seed=0
