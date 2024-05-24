#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH -J extract2D
#SBATCH -e extract2D.e%j
#SBATCH -o extract2D.o%j
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --constraint=HSW24

NB_NPROC=12 #(= 1 regions * 1 variable * 12 month)

runcode() { srun --mpi=pmi2 -m cyclic -n $@ ; }
liste=''

for reg in EATL; do
		for month in $(seq 1 12); do
			echo './compute_wbasemxl_NATL60_1month.ksh '$reg' CJM165 '$month >> tmp_compute_wbasemxl_NATL60-CJM165_m${month}.ksh
			chmod +x tmp_compute_wbasemxl_NATL60-CJM165_m${month}.ksh
			liste="$liste ./tmp_compute_wbasemxl_NATL60-CJM165_m${month}.ksh"
		done
done

runcode  $NB_NPROC /scratch/cnt0024/hmg2840/albert7a/bin/mpi_shell $liste

