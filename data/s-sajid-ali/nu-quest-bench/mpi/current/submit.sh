#!/bin/bash
#SBATCH --account=p30157
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=test_mpi
#SBATCH --output=outlog
#SBATCH --error=errlog

source /home/sas4990/packages/spack/share/spack/setup-env.sh
module purge all 

BASE_DIR=$(pwd)
OSU_VERSION=5.6.2

rm -rf osu-micro-benchmarks-5.6.2/
tar -xf osu-micro-benchmarks-5.6.2.tar.gz

while read -r mpimodule;
do
    echo "$mpiver" </dev/null

    mpiver=$(echo "$mpimodule" </dev/null | cut -c5-)
    cd ${BASE_DIR}/osu-micro-benchmarks-${OSU_VERSION}/
    mkdir build.$mpiver && cd build.$mpiver
    module load "$mpimodule" </dev/null

    mpiintelcheck=$(echo "$mpiver" </dev/null | cut -c1-5)

    if [ "$mpiintelcheck" = "intel" ]; then
	./../configure CC=mpiicc --prefix=$(pwd) &> "$mpiver.configlog"
    else
	./../configure CC=mpicc  --prefix=$(pwd) &> "$mpiver.configlog"
    fi

    cd mpi/one-sided/


    module list     &> "$mpiver.env"

    if [ "$mpiintelcheck" = "intel" ]; then
	mpiicc --version >> "$mpiver.env"
    else
	mpicc  --version >> "$mpiver.env"
    fi

    ld --version    >> "$mpiver.env" 

    mv ../../"$mpiver.configlog" .
    make          &> "$mpiver.makelog" 
    make install  &> "$mpiver.installlog" 

    export OMP_NUM_THREADS=1
    export MV2_ENABLE_AFFINITY=0

    for run in {1..20}
    do
        mpirun -np 2 ./osu_get_bw </dev/null &> bw_$run
        mpirun -np 2 ./osu_get_latency </dev/null &> lat_$run
    done

done < curr_mpi_list_sorted

