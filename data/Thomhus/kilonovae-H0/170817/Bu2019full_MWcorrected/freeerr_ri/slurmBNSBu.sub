#!/bin/bash
#SBATCH --job-name=BNSBu.job
#SBATCH --output=BNSBu%A_%a.out
#SBATCH --error=BNSBu%A_%a.err
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=thomas.hussenot@ijclab.in2p3.fr
#SBATCH -A umn131
#SBATCH --export=ALL
module purge
module load sdsc cpu/0.15.4 gcc/9.2.0 openmpi/4.1.1
module load anaconda3/2020.11
source /cm/shared/apps/spack/cpu/opt/spack/linux-centos8-zen2/gcc-10.2.0/anaconda3-2020.11-weucuj4yrdybcuqro5v3mvuq3po7rhjt/etc/profile.d/conda.sh
conda activate multinest
export LD_LIBRARY_PATH=/home/thussenot/MultiNest/lib/:$LD_LIBRARY_PATH
mpiexec -n 32 lightcurve-analysis --model Bu2019lm --svd-path /home/thussenot/nmma/svdmodels --outdir outdirBNSBu32cores --label AT170817 --prior ./Bu2019lm_AT170817.prior --tmin 0.01 --tmax 26 --dt 0.01  --nlive 2048 --Ebv-max 0 --trigger-time 57982.52851851852 --data ../../AT2017gfoMWcorrected.dat --plot --xlim 0,14 --ylim 26,16 --bestfit --filters ps1__r,ps1__i

