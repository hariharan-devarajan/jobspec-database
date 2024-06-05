#!/bin/bash
#SBATCH --account=p30157
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=4:00:00
#SBATCH --job-name=test_mpi
#SBATCH --output=outlog
#SBATCH --error=errlog

source /home/sas4990/miniconda3/etc/profile.d/conda.sh
conda activate ipy3

module purge all

BASE_DIR=$(pwd)
rm -rf questbeta
mkdir questbeta

while read -r namdmodule;
do
    cd $BASE_DIR

    namdver=$(echo "$namdmodule" </dev/null | cut -c6-)
    namdver=$(echo $namdver | sed "s/\//-/g")
    echo "$namdver" </dev/null

    module load "$namdmodule" </dev/null
    
    namdexec=$(python exec_dict.py $namdmodule </dev/null) </dev/null
    namdlaunch=$(python launch_dict.py $namdmodule </dev/null) </dev/null
 
    mkdir $BASE_DIR/questbeta/$namdver && cd $BASE_DIR/questbeta/$namdver
    cp $BASE_DIR/stmv.tar.gz .
    tar -xf stmv.tar.gz
    mv stmv/* .
    rm stmv.tar.gz
    rm -rf stmv/

    echo $namdexec

    module show "$namdmodule" </dev/null &> $namdmodule.modulefile
    ldd $(which $namdexec) </dev/null &> namdexec.link     

    export OMP_NUM_THREADS=1

    for run in {1..20}
    do
	    $namdlaunch +p48 ++mpiexec ++remote-shell mpiexec $namdexec stmv.namd </dev/null &> log_$run
	    ##$namdlaunch +p2 +ppn 24 ++mpiexec ++remote-shell mpiexec $namdexec stmv.namd  </dev/null &> log_$run
    done

done < curr_namd_list_sorted

