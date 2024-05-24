#!/bin/bash


### script to run several simulations with hybrid algorithms and different fitness
### landscapes on Orfeo, cluster hosted at Area Science Park (Trieste, Italy)
###
### number of genotipic classes = 4


#SBATCH --no-requeue
#SBATCH --job-name="fitness_comparison"
#SBATCH --partition=EPYC
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --time=02:00:00


module load architecture/AMD
module load conda/23.3.1

conda activate stoch_modelling

export OMP_NUM_THREADS=10


datafile=$(pwd)/data.csv

echo "# data for comparison between different fitness landscapes with" > "$datafile"
echo "# hybrid algorithm" >> "$datafile"
echo "#" >> "$datafile"
echo "# number of genotipic classes: 4" >> "$datafile"
echo "# population size: 10000" >> "$datafile"
echo "# " >> "$datafile"

echo "flat_simul,flat_elpsd,flat_state,static_inc_simul,static_inc_elpsd,static_inc_state,static_dec_simul,static_dec_elpsd,static_dec_state,static_mount_simul,static_mount_elpsd,static_mount_state,static_mount+dynamic_simul,static_mount+dynamic_elpsd,static_mount+dynamic_state" >> "$datafile"
for index in {1..20}
do
    python3 ../src/fixed_population.py --m 4 --N 10000 --fitness "flat" --output "final_state" --datafile "$datafile"
    echo -n "," >> "$datafile"
    python3 ../src/fixed_population.py --m 4 --N 10000 --fitness "static_inc" --output "final_state" --datafile "$datafile"
    echo -n "," >> "$datafile"
    python3 ../src/fixed_population.py --m 4 --N 10000 --fitness "static_dec" --output "final_state" --datafile "$datafile"
    echo -n "," >> "$datafile"
    python3 ../src/fixed_population.py --m 4 --N 10000 --fitness "static_mount" --output "final_state" --datafile "$datafile"
    echo -n "," >> "$datafile"
    python3 ../src/fixed_population.py --m 4 --N 10000 --fitness "dynamic" --output "final_state" --datafile "$datafile"
    echo >> "$datafile"
done 


module purge
