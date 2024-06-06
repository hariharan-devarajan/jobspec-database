#!/bin/bash
# Begin LSF directives
#BSUB -P MAT201 
#BSUB -J 8PSU_H_2000MVm_-x 
#BSUB -o std.oe%J
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags "gpumps smt1"
# End LSF directives and begin shell commands

[[ ${LSB_INTERACTIVE} == 'Y' ]] || cd ${LS_SUBCWD}
export ESPRESSO_PSEUDO=`readlink -f ~/Commons/UPF/polymer-pbe-van/`

source ~/env/lammps/module.sh

Ncpu=$((${LSB_MAX_NUM_PROCESSORS}-1))
Nnode=$((${Ncpu}/42))
Ngpu=$((${Nnode}*6))

set -x
Nmpi=${Ngpu}
N_rs=${Nmpi}
N__r=$((${N_rs}/${Nnode}))
N__a=$((${Nmpi}/${N_rs}))
N__g=$((${Ngpu}/${N_rs}))
N__c=$((${Ncpu}/${N_rs}))
N_kt=$((${N__c}/${N__a}))
N_kg=$((${N__g}/${N__a}))

my_lmp_mpi='../../bin/lmp'

jsrun -n${N_rs} -a${N__a} -c${N__c} -g${N__g} -r${N__r} \
      --bind none --latency_priority cpu-memory --smpiargs "-gpu" \
${my_lmp_mpi} -nocite -k on g ${N_kg} t ${N_kt} -sf kk -pk kokkos newton on \
 -i ./nvt.lmpi \
