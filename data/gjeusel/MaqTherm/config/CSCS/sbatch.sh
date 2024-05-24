#!/bin/bash -l
# -l will set modulecmd

if [ $# -lt 3 ] ; then 
   echo "USAGE : arg1=machine arg2=walltime arg3=job_np arg4=exe arg5=args_exe "
   echo "                       [arg6=mppnppn] [arg7=mppdepth]"
   echo "     [arg8=preaprun] [arg9=postaprun] [arg10=sbatchflags]"
   echo "     [arg11=outputpostfix] [arg12=hyperthreading]"
   exit 0
fi

cmd=`echo $0 "$@"`

cluster="$1"
walltime="$2"
job_np="$3"
exe="$4"
args_exe="$5"

if [ $cluster != "rothorn" ] && [ $cluster != "pilatus" ] && [ $cluster != "monch" ] ; then
        cpcn=`xtprocadmin -A |cut -c1-85| grep -m 1 compute|awk '{print $7}'`
        mppnppn="${6:-$cpcn}"
        if [ $mppnppn -gt $job_np ] ; then mppnppn=$job_np ; fi
        #if [ $mppnppn -lt $cpcn ] ; then mppnppn=$job_np ; fi
elif [ $cluster = "rothorn" ] ; then
        cpcn=256
        mppnppn="${6:-$cpcn}"

elif [ $cluster = "pilatus" ] ; then
        cpcn=32
        mppnppn="${6:-$cpcn}"
elif [ $cluster = "monch" ] ; then
        cpcn=40
        mppnppn="${6:-$cpcn}"
fi
# fi

mppdepth="${7:-1}"
# if [ -z $mppdepth ] ;then mppdepth=1 ; fi

numtasks=`expr $job_np \* $mppdepth | xargs printf "%04d\n"`
numtask=`echo $job_np | xargs printf "%d\n"`

preaprun=$8

postaprun=$9

sbatchflags=${10}
postfix=${11}
hyperthreading=${12}
if [ -z $hyperthreading ] ; then hyperthreading=1 ; fi
if [ $cluster = "santis" ] || [ $cluster = "daint" ] || [ $cluster = "brisi" ] || [ $cluster = "dora" ]; then
        ht1="#SBATCH --ntasks-per-core=$hyperthreading # -j"
        ht2="-j $hyperthreading"
fi

#echo "$job_np/$mppnppn"
cnodes=`perl -e "use POSIX qw(ceil);printf \"%d\n\",ceil($job_np/$mppnppn)"`
#cnodes=`awk -v n=$job_np -v N=$mppnppn 'function ceiling(x){return (x == int(x)) ? x : int(x)+1 }BEGIN{print ceiling($n/$N)}'`
#cnodes=`ceiling`
#echo "job_np=$job_np mppnppn=$mppnppn mppdepth=$mppdepth cpcn=$cpcn @ cnodes=$cnodes"
#exit 0
#==========================> cnodes=`ceiling`
oexe=`basename $exe`
out=runme.slurm.$cluster


 #####  ######     #    #     #
#     # #     #   # #    #   #
#       #     #  #   #    # #
#       ######  #     #    #
#       #   #   #######    #
#     # #    #  #     #    #
 #####  #     # #     #    #
if [ $cluster != "rothorn" ] && [ $cluster != "pilatus" ] && [ $cluster != "monch" ] ; then
cat <<EOF > $out
#!/bin/bash
##SBATCH --nodes=$cnodes
#
#SBATCH --ntasks=$numtask               # -n
#SBATCH --ntasks-per-node=$mppnppn      # -N
#SBATCH --cpus-per-task=$mppdepth       # -d
$ht1    # --ntasks-per-core / -j"
#
#SBATCH --time=00:$walltime:00
#SBATCH --job-name="nextFlow"
#SBATCH --output=o_$oexe.$numtasks.$mppnppn.$mppdepth.$hyperthreading.$cluster.$postfix
#SBATCH --error=o_$oexe.$numtasks.$mppnppn.$mppdepth.$hyperthreading.$cluster.$postfix
##SBATCH --account=usup
####SBATCH --reservation=maint

echo '# -----------------------------------------------'
ulimit -c unlimited
ulimit -s unlimited
ulimit -a |awk '{print "# "\$0}'
echo '# -----------------------------------------------'

echo '# -----------------------------------------------'
# export MPICH_CPUMASK_DISPLAY=1        # = core of the rank
# The distribution of MPI tasks on the nodes can be written to the standard output file by setting environment variable 
# export MPICH_RANK_REORDER_DISPLAY=1   # = node of the rank
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=536870912
export OMP_NUM_THREADS=$mppdepth
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
#export PAT_RT_CALLSTACK_BUFFER_SIZE=50000000 # > 4194312
#export OMP_STACKSIZE=500M
#
# export PAT_RT_EXPFILE_MAX=99999
# export PAT_RT_SUMMARY=0
#
#export PAT_RT_TRACE_FUNCTION_MAX=1024 
#export PAT_RT_EXPFILE_PES
#export MPICH_PTL_MATCH_OFF=1
#export MPICH_PTL_OTHER_EVENTS=4096
#export MPICH_MAX_SHORT_MSG_SIZE=32000
#export MPICH_PTL_UNEX_EVENTS=180000
#export MPICH_UNEX_BUFFER_SIZE=284914560
#export MPICH_COLL_OPT_OFF=mpi_allgather
#export MPICH_COLL_OPT_OFF=mpi_allgatherv
export MPICH_NO_BUFFER_ALIAS_CHECK=1
#NEW export MPICH_MPIIO_STATS=1
echo '# -----------------------------------------------'


echo '# -----------------------------------------------'
echo "# SLURM_JOB_NODELIST = \$SLURM_JOB_NODELIST"
echo "# submit command : \"$cmd\""
grep aprun $out
echo "# SLURM_JOB_NUM_NODES = \$SLURM_JOB_NUM_NODES"
echo "# SLURM_JOB_ID = \$SLURM_JOB_ID"
echo "# SLURM_JOBID = \$SLURM_JOBID"
echo "# SLURM_NTASKS = \$SLURM_NTASKS / -n --ntasks"
echo "# SLURM_NTASKS_PER_NODE = \$SLURM_NTASKS_PER_NODE / -N --ntasks-per-node"
echo "# SLURM_CPUS_PER_TASK = \$SLURM_CPUS_PER_TASK / -d --cpus-per-task"
echo "# OMP_NUM_THREADS = \$OMP_NUM_THREADS / -d "
echo "# SLURM_NTASKS_PER_CORE = \$SLURM_NTASKS_PER_CORE / -j1 --ntasks-per-core"
# sacct --format=JobID,NodeList%100 -j \$SLURM_JOB_ID
echo '# -----------------------------------------------'


date
set +x
/usr/bin/time -p $preaprun aprun -n $job_np -N $mppnppn -d $mppdepth $ht2 $postaprun $exe $args_exe 
# mv wave_tank1.h5 $job_np.wave_tank1.h5



EOF
fi


######    ###   #          #    ####### #     #  #####
#     #    #    #         # #      #    #     # #     #
#     #    #    #        #   #     #    #     # #
######     #    #       #     #    #    #     #  #####
#          #    #       #######    #    #     #       #
#          #    #       #     #    #    #     # #     #
#         ###   ####### #     #    #     #####   #####
if [ $cluster = "pilatus" ] || [ $cluster = "monch" ] ; then
mempilatus=`expr $numtasks \* 4000` # MB
cat <<EOF > $out
#!/bin/bash
##SBATCH --account=usup
#SBATCH -N $cnodes 
#SBATCH -n $job_np
#SBATCH --ntasks-per-node=$mppnppn
#SBATCH --cpu_bind=verbose
#SBATCH --time=00:$walltime:00
#SBATCH --job-name="jg"
#SBATCH --output=o_$oexe.$numtasks.$job_np.$mppnppn.$mppdepth.$cluster
#SBATCH --error=o_$oexe.$numtasks.$job_np.$mppnppn.$mppdepth.$cluster
##SBATCH --mem=63000 # MB

##SBATCH --nodes=$cnodes 
##SBATCH --threads-per-core=1
##SBATCH --cpus-per-task=$mppdepth
# -V
# --cpus-per-task | mppdepth
# --ntasks | job_np
#  --ntasks-per-core=<ntasks>
#  --ntasks-per-socket=<ntasks>
#  --ntasks-per-node=<ntasks>
# -----------------------------------------------------------------------
pwd
unset mc
ulimit -s unlimited

# -----------------------------------------------------------------------
# export MV2_ENABLE_AFFINITY=NO                                                                                
# *** Intel Only ***
# export KMP_AFFINITY=scatter,verbose
# export KMP_AFFINITY=compact,verbose
# -----------------------------------------------------------------------
# echo  "Running on nodes $SLURM_JOB_NODELIST"
export MPICH_CPUMASK_DISPLAY=1
# export MALLOC_MMAP_MAX_=0
# export MALLOC_TRIM_THRESHOLD_=536870912
export OMP_NUM_THREADS=$mppdepth
export MPICH_VERSION_DISPLAY=1
export MPICH_ENV_DISPLAY=1
#export PAT_RT_CALLSTACK_BUFFER_SIZE=50000000 # > 4194312

# export PAT_RT_EXPFILE_MAX=99999
# export PAT_RT_SUMMARY=0
#
#export PAT_RT_TRACE_FUNCTION_MAX=1024 
#export PAT_RT_EXPFILE_PES
#export MPICH_PTL_MATCH_OFF=1
#export MPICH_PTL_OTHER_EVENTS=4096
#export MPICH_MAX_SHORT_MSG_SIZE=32000
#export MPICH_PTL_UNEX_EVENTS=180000
#export MPICH_UNEX_BUFFER_SIZE=284914560

# ldd $exe |grep intel 2> /dev/null
# if [ \$? -eq 0 ] ; then 
#         # intel executables
#         export KMP_AFFINITY=disabled 
# else
#         export KMP_AFFINITY=enabled
#         #export GOMP_CPU_AFFINITY="\$cpuaff"
# fi
echo "SLURM_JOB_NAME=\$SLURM_JOB_NAME" "SLURM_JOBID=\$SLURM_JOBID SLURM_JOB_ID=\$SLURM_JOB_ID SLURM_TASK_PID=\$SLURM_TASK_PID OMP_NUM_THREADS=\$OMP_NUM_THREADS KMP_AFFINITY=\$KMP_AFFINITY VT_BUFFER_SIZE=\$VT_BUFFER_SIZE VT_MAX_FLUSHES=\$VT_MAX_FLUSHES VT_MODE=\$VT_MODE"

set +x
if [ $cluster = "monch" ] ; then
        isintelmpi=`which mpiexec |grep -q     impi ; echo $?`
         isopenmpi=`which mpiexec |grep -q  openmpi ; echo $?`
         ismvapich=`which mpiexec |grep -q mvapich2 ; echo $?`
        if [ \$isintelmpi = 0 ] ; then time -p $preaprun mpirun  -rmk slurm                       $postaprun $exe $args_exe ; fi
        if [ \$isopenmpi  = 0 ] ; then time -p $preaprun mpiexec -np $job_np -npernode $mppnppn $postaprun $exe $args_exe ; fi
        if [ \$ismvapich  = 0 ] ; then time -p $preaprun mpiexec -np $job_np -ppn      $mppnppn $postaprun $exe $args_exe ; fi
else
        time -p $preaprun srun $postaprun $exe $args_exe 
fi
#scan get confused : /usr/bin/time -p $preaprun srun --nodes=$cnodes --ntasks=$job_np --ntasks-per-node=$mppnppn $postaprun $exe $args_exe 
date +%D:%Hh%Mm%S

# ~/sbatch.sh pilatus ./a.out 1 4 4 8 # = 1min 4mpi*8omp 4mpi/1cn=1cn 
# ~/sbatch.sh pilatus ./a.out 1 4 1 8 # = 1min 4mpi*8omp 1mpi/1cn=4cn 

EOF
fi

 #####  ######     #    #######  #####  #     #
#     # #     #   # #      #    #     # #     #
#       #     #  #   #     #    #       #     #
 #####  ######  #     #    #    #       #######
      # #     # #######    #    #       #     #
#     # #     # #     #    #    #     # #     #
 #####  ######  #     #    #     #####  #     #

sbatch $sbatchflags $out
grep -E "aprun|mpirun|srun" $out |grep -v echo
exit 0
