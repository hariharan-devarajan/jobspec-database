#!/bin/bash
#SBATCH -p fpga 
#SBATCH --constraint=20.4.0_max 
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 2:00:00 
#SBATCH --fpgalink="n00:acl1:ch0-n01:acl0:ch1"
#SBATCH --fpgalink="n00:acl1:ch2-n01:acl0:ch3"
#SBATCH --fpgalink="n00:acl1:ch1-n01:acl0:ch0"
#SBATCH --fpgalink="n00:acl1:ch3-n01:acl0:ch2"
#SBATCH --fpgalink="n00:acl0:ch0-n00:acl0:ch1"
#SBATCH --fpgalink="n00:acl0:ch2-n00:acl0:ch3"
#SBATCH --fpgalink="n01:acl1:ch1-n01:acl1:ch0"
#SBATCH --fpgalink="n01:acl1:ch3-n01:acl1:ch2" 
#SBATCH -o ptrans_iec_N2_m64-%j.txt
#SBATCH -e ptrans_iec_N2_m64-%j.txt
#SBATCH -J PTRANS

module load intel intelFPGA_pro/21.2.0 bittware_520n/20.4.0_max devel/CMake/3.15.3-GCCcore-8.3.0

srun ../../synthesis_artifacts/PTRANS/520n-21.2.0-20.4.0-iec/Transpose_intel \
    -f ../../synthesis_artifacts/PTRANS/520n-21.2.0-20.4.0-iec/transpose_PQ_IEC.aocx \
    -n 10 -m 128 -b 512 -r 4
