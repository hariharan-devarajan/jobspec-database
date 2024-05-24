#!/bin/bash

module purge

# Main OmpSs-2 utilities used in Examples and Exercises:
module load intel/2017.2
module load gcc/6.2.0
module load impi/2017.1
module load mkl/2017.2
module load bsc/current
module load ompss-2/2019.11.2
module load tampi/1.0.1
module load extrae
module load paraver

# Job scheduler header for one node execution
cat > $OSS2EE_CONFIG_DIR/job-sched-1n << EOF
#!/bin/bash
#SBATCH --job-name=ompss-2-ee
#SBATCH --workdir=.
#SBATCH --output=oss-2-ee_%j.out
#SBATCH --error=oss-2-ee_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --qos=debug
EOF

# Job scheduler header for multi-node execution
cat > $OSS2EE_CONFIG_DIR/job-sched-mn << EOF
#!/bin/bash
#SBATCH --job-name=oss-2-ee
#SBATCH --workdir=.
#SBATCH --output=oss-2-ee_%j.out
#SBATCH --error=oss-2-ee_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=2
#SBATCH --time=00:15:00
#SBATCH --qos=debug
EOF

# Extra package configuration (building purposes): MPI
export OSS2EE_MPI_LIB_DIR=/apps/INTEL/2017.2-028/impi/2017.2.191/lib64
export OSS2EE_MPI_INC_DIR=/apps/INTEL/2017.2-028/impi/2017.2.191/include64

# Extra package configuration (building purposes): MKL
export OSS2EE_MKL_LIB_DIR=/apps/INTEL/2017.2-028/mkl/lib/intel64/
export OSS2EE_MKL_INC_DIR=/apps/INTEL/2017.2-028/mkl/include

# Extra package configuration (building purposes): ATLAS
export OSS2EE_ATL_LIB_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/lib
export OSS2EE_ATL_INC_DIR=/gpfs/apps/NVIDIA/ATLAS/3.9.51/include/

# Extra package configuration (for building purposes): TAMPI
export OSS2EE_TAMPI_LIB_DIR=$TAMPI_HOME/lib
export OSS2EE_TAMPI_INC_DIR=$TAMPI_HOME/include

# Command for compiling and running hybrid MPI programs
export OSS2EE_MPICC_COMMAND="mpiicc"
export OSS2EE_MPICXX_COMMAND="mpiicpc"
export OSS2EE_MPIRUN_COMMAND="srun --cpu_bind=cores"

# Put all Nanos6 feature by default
unset NANOS6_CONFIG_OVERRIDE
