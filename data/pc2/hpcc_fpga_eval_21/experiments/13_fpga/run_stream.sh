#!/bin/bash
#
#SBATCH -p fpga
#SBATCH -J stream
#SBATCH --constraint=20.4.0_hpc
#SBATCH -N 13
#SBATCH -n 26
#SBATCH -t 2:00:00
#SBATCH -o stream_N13-%j.out
#SBATCH -e stream_N13-%j.out
#
##
#
# Execute the benchmarks synthesized with the HPC BSP on 4 FPGAs.
# The used FPGA is specified in the whitelist (-w) above.
# Results are stored in separate files for each benchmark
#

# Preparations
module load intelFPGA_pro/21.2.0 bittware_520n/20.4.0_hpc intel devel/CMake/3.15.3-GCCcore-8.3.0 

# Execute STREAM with 4GB arrays = 12 GB per FPGA, 48GB total
# Same inputs as 1 FPGA because the benchmark is EP
srun ../../synthesis_artifacts/STREAM/520n-21.2.0-20.4.0/STREAM_FPGA_intel -f ../../synthesis_artifacts/STREAM/520n-21.2.0-20.4.0/stream_kernels_single.aocx -r 4 -s 1073741824


