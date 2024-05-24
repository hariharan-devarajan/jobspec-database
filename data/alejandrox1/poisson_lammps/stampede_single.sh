#!/bin/bash
#SBATCH --exclusive           # Individual nodes
#SBATCH -t 2-0                # Run time (hh:mm:ss)
#SBATCH -o slurm.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alarcj137@gmail.com

#~
#~	in.elastic calls	
#~
#~	init.mod : 	Input sturcture, lattice parameters and units.
#~	potential.mod :	Desired force field.
#~	displace.mod
#~

module load lammps

procs=$1
latttype=$2
lattconst=$3
deform=$4
struct=$5

# Determine the host
host=$(hostname)
set -- "$host"
IFS="."; declare -a array=($*)
echo "${array[@]}"
flag="${array[1]}"
IFS=" "

case "$flag" in
        "stampede")
                cluster="$flag"
                partition=normal
                cpern=16

                echo "Running on sampede.tacc.xsede.org"

                runfiles=/home1/03561/alarcj/LAMMPS
                structures=/home1/03561/alarcj/LAMMPS/schwarzite/atomic
                cp ${structures}/${struct} .
                cp ${runfiles}/in.elastic .
                cp ${runfiles}/init.mod .
                cp ${runfiles}/potential.mod .
                cp ${runfiles}/displace.mod .
                cp ${runfiles}/CH.airebo . 
                ;;
        "sdsc")
                cluster=comet
                files=/oasis/scratch/comet/alarcj/temp_project/AMINO/data/op_run
                partition=compute
                echo "Running on comet.sdsc.xsede.org"
                ;;
        *)
                echo "Unkown option."
                exit
esac

# finite deformation size (1.0e-6)
#deform in 1.0e-5 5.0e-5 1.0e-6 5.0e-6 1.0e-7; do

# Modify the pair_coeff with respect to the number of atom types
types=`awk 'FNR == 7 {print $1}' ${struct}`
string=$(for ((i=1; i<=$types; i++)); do printf "%s" " C "; done)
sed -i "s/CH.airebo C.*/CH.airebo ${string}/" potential.mod

# For rediretion '| tee filename'
ibrun -np $procs lmp_stampede -var fname ${struct} -var up ${deform} -var lattype ${latttype} -var latconst ${lattconst} < in.elastic 
		


