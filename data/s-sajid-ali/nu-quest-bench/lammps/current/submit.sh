#!/bin/bash
#SBATCH --account=p30157
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=2:00:00
#SBATCH --job-name=test_mpi
#SBATCH --output=outlog
#SBATCH --error=errlog

source /home/sas4990/miniconda3/etc/profile.d/conda.sh
conda activate ipy3

module purge all

BASE_DIR=$(pwd)
rm -rf questbeta
mkdir questbeta

while read -r lammpsmodule;
do
    cd $BASE_DIR

    lammpsver=$(echo "$lammpsmodule" </dev/null | cut -c8-)
    lammpsver=$(echo $lammpsver | sed "s/\//-/g")
    echo "$lammpsver" </dev/null

    module load "$lammpsmodule" </dev/null
    
    lammpsexec=$(python exec_dict.py $lammpsmodule </dev/null) </dev/null
 
    mkdir $BASE_DIR/questbeta/$lammpsver && cd $BASE_DIR/questbeta/$lammpsver
    cp $BASE_DIR/in.lj .

    echo $lammpsexec

    module show "$lammpsmodule" </dev/null &> $lammpsmodule.modulefile
    ldd $(which $lammpsexec) </dev/null &> $lammpsexec.link     

    export OMP_NUM_THREADS=1

    if [$lammpsmodule = "lammps/lammps-10Feb2015/lammps-gcc"];
    then
	    export OMPI_MCA_rmaps_base_oversubscribe=1
    fi

    if [$lammpsmodule = "lammps/lammps-22Aug18"];
    then
	    export OMPI_MCA_rmaps_base_oversubscribe=1
    fi

    for run in {1..20}
    do
	    mpirun -np 48 $lammpsexec -in in.lj  </dev/null &> log_$run
    done

done < curr_lammps_list_sorted

