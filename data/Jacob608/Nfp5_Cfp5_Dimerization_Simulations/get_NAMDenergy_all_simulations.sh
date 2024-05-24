#!/bin/bash
#SBATCH --job-name="Collect NAMD Energy"
#SBATCH -A p31412
#SBATCH -p short    ## partition
#SBATCH -N 1  ## number of nodes
#SBATCH --ntasks-per-node=1  ## number of cores
#SBATCH -t 4:00:00

# Load necessary modules.
module load vmd

# Load all the simulation names stored in file names.txt into the list names.
names=()
while IFS= read -r line; do
	names+=("$line")
done < "names.txt"

# For each element in names, navigate into that directory, copy get_NAMDenergy.vmd into that directory, and execute get_NAMDenergy.vmd in VMD.
for element in "${names[@]}"; do
	cd $element
	cp ../get_NAMDenergy.vmd .
	vmd -dispdev text -e get_NAMDenergy.vmd >> get_NAMDenergy.log
	cd ..
done
