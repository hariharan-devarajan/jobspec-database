#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -J task1_thrust
#SBATCH -o %x.out -e %x.err
#SBATCH -t 0-00:02:00
#SBATCH --gres=gpu:1
module load nvidia/cuda/11.8.0 
nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_thrust
for i in {10..20}
do
	./task1_thrust $((2**i)) 
done
for i in {0..10}
do
        if [ $i -eq 0 ]; then
                sed -n "$((2*i+2))p" task1_thrust.out > task1c_thrust.log
        else
                sed -n "$((2*i+2))p" task1_thrust.out >> task1c_thrust.log
        fi
done