#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -n 1
#SBATCH -q regular
#SBATCH -J ARTMS
#SBATCH --mail-type=ALL
#SBATCH -t 12:00:00


# OpenMP settings:
# export OMP_NUM_THREADS=64
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

#run the application: 
#module load python
#conda activate Titania
# dry run of snakefile with thread command

module load python

# check if conda environment exists

mamba env create -n Titania -f envs/Titania.yaml

conda activate Titania

# Check for 4CAC

DIR="./resources/eukccdb"
if [ -d $DIR ]
then
    echo "eukccdb directory exists"
    cd resources/eukccdb
    export EUKCC2_DB=$(realpath eukcc2_db_ver_1.1)
    cd ../..

else
    echo "eukccdb directory does not exist"
    echo "Fetching eukccdb directory"
    mkdir resources/eukccdb
    cd resources/eukccdb
    wget http://ftp.ebi.ac.uk/pub/databases/metagenomics/eukcc/eukcc2_db_ver_1.1.tar.gz
    tar -xzvf eukcc2_db_ver_1.1.tar.gz
    export EUKCC2_DB=$(realpath eukcc2_db_ver_1.1)
    cd ../..
fi

DIR="./resources/4CAC"
if [ -d $DIR ]
then
    echo "4CAC directory exists"
else
    echo "4CAC directory does not exist"
    echo "Fetching 4CAC directory"
    git clone https://github.com/Shamir-Lab/4CAC.git
    mv 4CAC/ resources/.
fi

snakemake --cores 240 --use-conda --snakefile rules/mag_stats.smk