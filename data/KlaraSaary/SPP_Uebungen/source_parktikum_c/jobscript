#!/bin/bash
#SBATCH -J my_job
#SBATCH --mail-user=felix.staniek@gmx.de
#SBATCH --mail-type=ALL
#SBATCH -e Job_name.err.%j
#SBATCH -o Job_name.out.%j
#SBATCH --mem-per-cpu=1800
#SBATCH -t 00:03:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -p kurs1
#SBATCH --exclusive

echo "This is Job $SLURM_JOB_ID"
module load gcc
##module load openmpi/gcc
cd /home/kurse/kurs1/ui31dymo/Lap1/SPP_Uebungen/source_parktikum_c
./main text1.txt text4.txt

