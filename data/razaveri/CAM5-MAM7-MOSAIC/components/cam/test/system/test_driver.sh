#!/bin/sh -f
#
# test_driver.sh:  driver script for the testing of CAM in Sequential CCSM
#
# usage on hobart, leehill, firefly, yellowstone, jaguarpf/titan, lynx:
# ./test_driver.sh
#
# valid arguments: 
# -i    interactive usage
# -f    force batch submission (avoids user prompt)
# -j    number of jobs for gmake
# -b    support baseline scripts for cam5_2_12 and earlier.
# -h    displays this help message
#
# **pass environment variables by preceding above commands 
#   with 'env var1=setting var2=setting '
# **more details in the CAM testing user's guide, accessible 
#   from the CAM developers web page

interactive=false
force=false
gmake_j=8

while [ -n "$(echo $1 | grep '-')" ]; do
    case $1 in

	-i ) interactive=true ;;

	-f ) force=true 
             if  $interactive ; then
               echo "test_driver.sh: FATAL ERROR: -i and -f were set"
             exit 1
             fi
             ;;

	-j ) shift; gmake_j=$1 ;;

        -b ) export CAM_BASEBACK="YES"
             ;; 

	-h ) echo 'usage: test_driver.sh [-i] [-f] [-h] [-j N] [-b]...'; exit 1 ;;

	* ) echo "test_driver.sh: FATAL ERROR: unrecognized arg: $1"; exit 1 ;;

    esac
    shift
done


#will attach timestamp onto end of script name to prevent overwriting
cur_time=`date '+%H:%M:%S'`

hostname=`hostname`

case $hostname in

    ##yellowstone (or caldera)
    ye* | ys* | ca* )
    submit_script="`pwd -P`/test_driver_yellowstone_${cur_time}.sh"
    submit_script_cb="`pwd -P`/test_driver_yellowstone_cb_${cur_time}.sh"

    if [ -z "$CAM_ACCOUNT" ]; then
        export CAM_ACCOUNT=`grep -i "^${LOGNAME}:" /glade/u/ssg/ys/opt/lsf/conf/configdir/project.ncar | cut -f 1 -d "," | cut -f 2 -d ":" `
        if [ -z "${CAM_ACCOUNT}" ]; then
            echo "ERROR: unable to locate an account number to charge for this job under user: $LOGNAME"
            exit 2
        fi
    fi
 
    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="regular"
    fi

    if [ -z "$CALDERA_BATCHQ" ]; then
       export CALDERA_BATCHQ="caldera"
    fi

    if [ "$CALDERA_BATCHQ" = "gpgpu" ]; then
      exclsnode="#BSUB -x"
      cald_nodes="#BSUB -n 32"                          # exclusive use of node (not_shared)
    else
      cald_nodes="#BSUB -n 15"                          # exclusive use of node (not_shared)
    fi

    if [ "$CAM_BATCHQ" = "small" ]; then
        wallclock_limit="2:00"
    else
        wallclock_limit="8:00"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=16
    fi

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=32
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( 2 * $CAM_TASKS ))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script_cb} << EOF
#!/bin/sh
#

#BSUB -a poe                    # use LSF poe elim
${exclsnode}
${cald_nodes}
#BSUB -R "span[ptile=32]"          # max number of tasks (MPI) per node
#BSUB -o test_dr.o%J              # output filename
#BSUB -e test_dr.o%J              # error filename
#BSUB -q $CALDERA_BATCHQ
#BSUB -W 6:00
#BSUB -P $CAM_ACCOUNT    
#BSUB -J test_dr
if [ -n "\$LSB_JOBID" ]; then   #batch job
    export JOBID=\${LSB_JOBID}
    initdir=\${LS_SUBCWD}
    interactive=false
else
    interactive=true
fi

if [ -z "$CAM_RBOPTIONS" ]; then
    export CAM_RBOPTIONS="build_only"
fi

## create_newcase looks for account number in ACCOUNT environment variable
export ACCOUNT=$CAM_ACCOUNT

##omp threads
export CAM_THREADS=2
export CAM_RESTART_THREADS=1

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

# This is a floor preventing the use of 32 MPI tasks per node, which is
# frequently broken, but allowing 32 threads.
export min_cpus_per_task=1

source /glade/apps/opt/lmod/lmod/init/bash
module purge

if [ "\$CAM_FC" = "PGI" ]; then
  module load ncarenv/1.0
  module load ncarbinlibs/1.0
  module load pgi/12.5
  module load ncarcompilers
  module load netcdf/4.2
  module load pnetcdf/1.3.0
  module load perlmods
  export CFG_STRING=" -cc mpicc -fc_type pgi -fc mpif90 -cppdefs -DNO_MPI2 -cppdefs -DNO_MPIMOD "
else
  module load intel/15.0.3
  module load ncarenv/1.0
  module load ncarbinlibs/1.0
  module load ncarcompilers/1.0
  module load netcdf-mpi/4.3.0
  module load pnetcdf/1.3.0
  module load mkl/11.0.1
  module load perlmods
  export CFG_STRING="-cc mpicc -fc mpif90 -fc_type intel "
fi

export MAKE_CMD="gmake -j $gmake_j"
export CCSM_MACH="yellowstone"
export MACH_WORKSPACE="/glade/scratch"
export CPRNC_EXE=${CESMDATAROOT}/tools/cprnc/cprnc
export ADDREALKIND_EXE=/fs/cgd/csm/tools/addrealkind/addrealkind
dataroot=${CESMDATAROOT}
echo_arg="-e"
input_file="tests_pretag_yellowstone"

EOF

#-------------------------------------------

cat > ${submit_script} << EOF
#!/bin/sh
#

#BSUB -a poe                     # use LSF poe elim
#BSUB -x                          # exclusive use of node (not_shared)
#BSUB -n 64                       # total tasks needed
#BSUB -R "span[ptile=16]"          # max number of tasks (MPI) per node
#BSUB -o test_dr.o%J              # output filename
#BSUB -e test_dr.o%J              # error filename
#BSUB -q $CAM_BATCHQ
#BSUB -W $wallclock_limit
#BSUB -P $CAM_ACCOUNT   
#BSUB -J test_dr
if [ -n "\$LSB_JOBID" ]; then   #batch job
    export JOBID=\${LSB_JOBID}
    initdir=\${LS_SUBCWD}
    interactive=false
else
    interactive=true
fi

if [ -z "$CAM_RBOPTIONS" ]; then
    export CAM_RBOPTIONS="run_only"
fi

##omp threads
export CAM_THREADS=2
export CAM_RESTART_THREADS=1

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

# This is a floor preventing the use of 32 MPI tasks per node, which is
# frequently broken, but allowing 32 threads.
export min_cpus_per_task=1

source /glade/apps/opt/lmod/lmod/init/bash
module purge

if [ "\$CAM_FC" = "PGI" ]; then
  module load ncarenv
  module load ncarbinlibs
  module load pgi
  module load ncarcompilers
  module load netcdf
  module load pnetcdf
  module load perlmods
  export CFG_STRING=" -cc mpicc -fc_type pgi -fc mpif90 -cppdefs -DNO_MPI2 -cppdefs -DNO_MPIMOD "
else
  module load intel/15.0.1
  module load ncarenv/1.0
  module load ncarbinlibs/1.0
  module load ncarcompilers/1.0
  module load netcdf-mpi/4.3.0
  module load pnetcdf/1.3.0
  module load mkl/11.0.1
  module load perlmods
  export CFG_STRING="-cc mpicc -fc mpif90 -fc_type intel "
fi



#export MP_MPILIB=\${MPILIB}
#CAF export INC_NETCDF=\${NETCDF}/include
#CAF export LIB_NETCDF=\${NETCDF}/lib
#export INC_PNETCDF=/contrib/parallel-netcdf-1.2.0-svn961/include
#export LIB_PNETCDF=/contrib/parallel-netcdf-1.2.0-svn961/lib
export MAKE_CMD="gmake -j $gmake_j"
export CCSM_MACH="yellowstone"
export MACH_WORKSPACE="/glade/scratch"
export CPRNC_EXE=${CESMDATAROOT}/tools/cprnc/cprnc
export ADDREALKIND_EXE=/fs/cgd/csm/tools/addrealkind/addrealkind
dataroot=${CESMDATAROOT}
echo_arg="-e"
input_file="tests_pretag_yellowstone"
#input_file="mylist"

EOF

##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;

    ##hobart
    hob* | h[[:digit:]]* )
    submit_script="`pwd -P`/test_driver_hobart_${cur_time}.sh"
    export PATH=/cluster/torque/bin:${PATH}

    # Default setting is 12 hr in the long queue; the short queue only
    # allows 1 hr runs.
    wallclock_limit="12:00:00"
    gmake_j=24
    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="long"
    elif [[ "$CAM_BATCHQ" == short ]]; then
        wallclock_limit="1:00:00"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=24
    fi

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=48
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( $CAM_TASKS / 2))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Name of the queue (CHANGE THIS if needed)
#PBS -q $CAM_BATCHQ
# Number of nodes (CHANGE THIS if needed)
#PBS -l walltime=$wallclock_limit,nodes=1:ppn=48
# output file base name
#PBS -N test_dr
# Put standard error and standard out in same file
#PBS -j oe
# Export all Environment variables
#PBS -V
# End of options

# Make sure core dumps are created
ulimit -c unlimited

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

if [ -z "$CAM_RBOPTIONS" ]; then
    export CAM_RBOPTIONS="run_and_build"
fi

##omp threads
export CAM_THREADS=1
export CAM_RESTART_THREADS=2

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

export P4_GLOBMEMSIZE=500000000
source /usr/share/Modules/init/sh
module purge

export LAPACK_LIBDIR=/usr/lib64

# Not currently used on hobart
#/usr/local/bin/make_ib_hosts.sh

if [ "\$CAM_FC" = "INTEL" ]; then
    module load compiler/intel/14.0.2
    export CFG_STRING=" -cc mpicc -fc_type intel -fc mpif90 -cppdefs -DNO_MPI2 -cppdefs -DNO_MPIMOD "
    export INC_NETCDF=\${NETCDF_PATH}/include
    export LIB_NETCDF=\${NETCDF_PATH}/lib
    input_file="tests_pretag_hobart_nag"
    export CCSM_MACH="hobart_intel"
elif [ "\$CAM_FC" = "NAG" ]; then
    module load compiler/nag/6.0
    module load mpi/nag/mvapich2-1.8.1-qlc
    export CFG_STRING="-cc mpicc -fc mpif90 -fc_type nag "
    export INC_NETCDF=\${NETCDF_PATH}/include
    export LIB_NETCDF=\${NETCDF_PATH}/lib
    input_file="tests_pretag_hobart_nag"
    export CCSM_MACH="hobart_nag"
else
    module load compiler/pgi/15.1
#    module load mpi/pgi/mvapich2-1.8.1-qlc
    export CFG_STRING=" -cc mpicc -fc_type pgi -fc mpif90 -cppdefs -DNO_MPI2 -cppdefs -DNO_MPIMOD "
    export INC_NETCDF=\${NETCDF_PATH}/include
    export LIB_NETCDF=\${NETCDF_PATH}/lib
    input_file="tests_pretag_hobart_pgi"
    export CCSM_MACH="hobart_pgi"
fi
export MAKE_CMD="gmake --output-sync -j $gmake_j"
export MACH_WORKSPACE="/scratch/cluster"
export CPRNC_EXE=/fs/cgd/csm/tools/cprnc_hobart/cprnc
export ADDREALKIND_EXE=/fs/cgd/csm/tools/addrealkind/addrealkind
dataroot="/fs/cgd/csm"
echo_arg="-e"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;

    ##leehill - always run with CAM_FC=PGI and -i
    le* )
    submit_script="`pwd -P`/test_driver_leehill${cur_time}.sh"
    export PATH=/cluster/torque/bin:${PATH}

    # Default setting is 12 hr in the long queue; the short queue only
    # allows 1 hr runs.
    wallclock_limit="12:00:00"
    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="long"
    elif [[ "$CAM_BATCHQ" == short ]]; then
        wallclock_limit="1:00:00"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=8
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Make sure core dumps are created
ulimit -c unlimited

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

echo ${interactive}

if [ -z "$CAM_RBOPTIONS" ]; then
    export CAM_RBOPTIONS="run_and_build"
fi

export INTEL=/usr/local/intel-cluster
export NAG=/usr/local/nag
export PGI=/usr/local/pgi
export LD_LIBRARY_PATH=\${PGI}/linux86-64/lib:/cluster/torque/lib:\${INTEL}/lib/intel64
echo \${LD_LIBRARY_PATH}
export P4_GLOBMEMSIZE=500000000

# Only PGI is supported on leehill at this time
#if [ "\$CAM_FC" = "PGI" ]; then
    export LAPACK_LIBDIR=\${PGI}/linux86-64/lib
    export INC_NETCDF=/usr/local/netcdf-pgi/include
    export LIB_NETCDF=/usr/local/netcdf-pgi/lib
    export PATH=\${PGI}/linux86-64/bin:\${PATH}:\${LIB_NETCDF}
    export LD_LIBRARY_PATH=\${PGI}/linux86-64/libso:\${LIB_NETCDF}:\${LD_LIBRARY_PATH}
    export CFG_STRING=" -cc pgcc -fc_type pgi -fc pgf90 "
    export input_file="tests_pretag_leehill"
    export CCSM_MACH="leehill_pgi"
#fi
export MAKE_CMD="gmake -j $gmake_j"
export MACH_WORKSPACE="/scratch/cluster"
export CPRNC_EXE=/fs/cgd/csm/tools/cprnc_64/cprnc
export ADDREALKIND_EXE=/fs/cgd/csm/tools/addrealkind/addrealkind
dataroot="/fs/cgd/csm"
echo_arg="-e"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;

    ##edison
    ed* )
    submit_script="`pwd -P`/test_driver_edison_${cur_time}.sh"

    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="regular"
    fi

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=16
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( $CAM_TASKS / 2 ))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Name of the queue (CHANGE THIS if needed)
#PBS -q $CAM_BATCHQ
# Number of procs and walltime (CHANGE THIS if needed)
#PBS -l mppwidth=16,walltime=04:30:00
# output file base name
#PBS -N test_dr
# Put standard error and standard out in same file
#PBS -j oe
# Export all Environment variables
#PBS -V
# End of options

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

if [ -z "$CAM_RBOPTIONS" ]; then
    export CAM_RBOPTIONS="run_and_build"
fi
##omp threads
export CAM_THREADS=1
export CAM_RESTART_THREADS=2

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

. /opt/modules/default/init/sh
module load cray-netcdf

export CFG_STRING="-fc ftn -cc cc -fc_type intel "
export CCSM_MACH="edison_intel"
module list

export MPICH_MAX_SHORT_MSG_SIZE=1024

export MAKE_CMD="gmake -j 8"
export MACH_WORKSPACE="/scratch1/scratchdirs"
export CPRNC_EXE=/project/projectdirs/ccsm1/tools/cprnc/cprnc
export ADDREALKIND_EXE=/project/projectdir/ccsm1/tools/addrealkind/addrealkind
dataroot="/project/projectdirs/ccsm1"
echo_arg="-e"
input_file="tests_pretag_edison_intel"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;


    ##titan
    ti* | ti* )
    submit_script="`pwd -P`/test_driver_titanpf_${cur_time}.sh"

    if [ -z "$CAM_ACCOUNT" ]; then
	export CAM_ACCOUNT=`showproj -s jaguarpf | tail -1 `
	if [ -z "${CAM_ACCOUNT}" ]; then
	    echo "ERROR: unable to locate an account number to charge for this job under user: $LOGNAME"
	    exit 2
	fi
    fi

    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="batch"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=8
    fi

    machine="titan"

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=8
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( $CAM_TASKS / 2 ))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Name of the queue (CHANGE THIS if needed)
#PBS -q $CAM_BATCHQ
# Number of procs and walltime (CHANGE THIS if needed)
#PBS -l walltime=11:58:00,size=12
# output file base name
#PBS -N test_dr
# Put standard error and standard out in same file
#PBS -j oe
# Export all Environment variables
#PBS -V
#PBS -A $CAM_ACCOUNT
# End of options

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

##omp threads
export CAM_THREADS=1
export CAM_RESTART_THREADS=2

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

. /opt/modules/default/init/sh
module load netcdf
module switch pgi/10.9.0 pgi/11.8.0
module list

export INC_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pgi/include
export LIB_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pgi/lib

export MPICH_MAX_SHORT_MSG_SIZE=1024

export MAKE_CMD="gmake -j $gmake_j"
export CFG_STRING="-fc ftn "
export CCSM_MACH="$machine"
export MACH_WORKSPACE="/tmp/work"
export CPRNC_EXE=/tmp/proj/ccsm/tools/ccsm_cprnc/cprnc
export ADDREALKIND_EXE=/tmp/proj/ccsm/tools/addrealkind/addrealkind
dataroot="/tmp/proj/ccsm"
echo_arg="-e"
input_file="tests_posttag_titan"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;

    ##hopper
    hop* | ni* )
    submit_script="`pwd -P`/test_driver_hopper_${cur_time}.sh"

    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="regular"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=8
    fi

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=8
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( $CAM_TASKS / 2 ))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Name of the queue (CHANGE THIS if needed)
#PBS -q $CAM_BATCHQ
# Number of procs and walltime (CHANGE THIS if needed)
#PBS -l mppwidth=24,walltime=05:58:00
# output file base name
#PBS -N test_dr
# Put standard error and standard out in same file
#PBS -j oe
# Export all Environment variables
#PBS -V
# End of options

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

##omp threads
export CAM_THREADS=1
export CAM_RESTART_THREADS=2

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

. /opt/modules/default/init/sh
module unload PrgEnv-pathscale
module unload PrgEnv-intel
module unload PrgEnv-pgi
module load netcdf

if [ "\$CAM_FC" = "PATHSCALE" ]; then
    module load PrgEnv-pathscale
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/pathscale/109/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/pathscale/109/lib
    export CFG_STRING="-fc ftn -cc cc -linker ftn -fc_type pathscale "
    export CCSM_MACH="hopper_pathscale"
    export USER_CC="cc"
elif [ "\$CAM_FC" = "INTEL" ]; then
    module load PrgEnv-intel
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/intel/109/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/intel/109/lib
    export CFG_STRING="-fc ftn -cc cc -fc_type intel "
    export CCSM_MACH="hopper_intel"
else
    module load PrgEnv-pgi
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/pgi/109/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/pgi/109/lib
    export CFG_STRING="-fc ftn -cc cc -fc_type pgi "
    export CCSM_MACH="hopper_pgi"
fi
module list

export MPICH_MAX_SHORT_MSG_SIZE=1024

export MAKE_CMD="gmake -j $gmake_j"
export MACH_WORKSPACE="/scratch/scratchdirs"
export CPRNC_EXE=/project/projectdirs/ccsm1/tools/cprnc/cprnc
export ADDREALKIND_EXE=/project/projectdir/ccsm1/tools/addrealkind/addrealkind
dataroot="/project/projectdirs/ccsm1"
echo_arg="-e"
input_file="tests_pretag_hobart_pgi"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;


    ##lynx
    ly* )
    submit_script="`pwd -P`/test_driver_lynx_${cur_time}.sh"

    if [ -z "$CAM_BATCHQ" ]; then
        export CAM_BATCHQ="regular"
    fi

    if [ $gmake_j = 0 ]; then
        gmake_j=8
    fi

    if [ -z "$CAM_TASKS" ]; then
        CAM_TASKS=18
    fi
    if [ -z "$CAM_RESTART_TASKS" ]; then
        CAM_RESTART_TASKS=$(( $CAM_TASKS * 4 / 3 ))
    fi

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv
cat > ${submit_script} << EOF
#!/bin/sh
#

# Name of the queue (CHANGE THIS if needed)
#PBS -q $CAM_BATCHQ
# Number of procs and walltime (CHANGE THIS if needed)
#PBS -l mppwidth=72,walltime=05:58:00
# output file base name
#PBS -N test_dr
# Put standard error and standard out in same file
#PBS -j oe
# Export all Environment variables
#PBS -V
# #PBS -A $CAM_ACCOUNT
#PBS -A UT-NTNL0121
# End of options

if [ -n "\$PBS_JOBID" ]; then    #batch job
    export JOBID=\`echo \${PBS_JOBID} | cut -f1 -d'.'\`
    initdir=\${PBS_O_WORKDIR}
fi

if [ "\$PBS_ENVIRONMENT" = "PBS_BATCH" ]; then
    interactive=false
else
    interactive=true
fi

##omp threads
export CAM_THREADS=4
export CAM_RESTART_THREADS=3

##mpi tasks
export CAM_TASKS=$CAM_TASKS
export CAM_RESTART_TASKS=$CAM_RESTART_TASKS

. /opt/modules/default/init/sh
module unload PrgEnv-pathscale
module unload PrgEnv-intel
module unload PrgEnv-pgi
module unload pgi
module unload intel
module unload pathscale

module load netcdf

if [ "\$CAM_FC" = "PATHSCALE" ]; then
    module load PrgEnv-pathscale
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pathscale/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pathscale/lib
    export CFG_STRING="-fc ftn -cc cc -linker ftn -fc_type pathscale "
    export CCSM_MACH="lynx_pathscale"
    export USER_CC="cc"
elif [ "\$CAM_FC" = "INTEL" ]; then
    module load PrgEnv-intel
    module switch intel intel/12.1.0
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-intel/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-intel/lib
    export CFG_STRING="-fc ftn -cc cc -fc_type intel "
    export CCSM_MACH="lynx_intel"
else
    module load PrgEnv-pgi
    module switch pgi pgi/11.10.0
    export INC_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pgi/include
    export LIB_NETCDF=\${CRAY_NETCDF_DIR}/netcdf-pgi/lib
    export CFG_STRING="-fc ftn -cc cc -fc_type pgi "
    export CCSM_MACH="lynx_pgi"
fi
module list

export MPICH_MAX_SHORT_MSG_SIZE=1024

export MAKE_CMD="gmake -j $gmake_j"
export MACH_WORKSPACE="/ptmp"
export CPRNC_EXE=/glade/proj3/cseg/tools/cprnc.lynx/cprnc
export ADDREALKIND_EXE=/glade/proj3/cseg/tools/addrealkind/addrealkind
dataroot="/glade/proj3/cseg"
echo_arg="-e"
input_file="tests_posttag_titan"

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^
    ;;


    * ) echo "ERROR: machine $hostname not currently supported"; exit 1 ;;
esac

##vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv writing to batch script vvvvvvvvvvvvvvvvvvv

cb_flag=0


for file in ${submit_script} ${submit_script_cb}
do
cat >> ${file} << EOF

## Most machines don't need to set this, so default to 1.
if [[ -z "\$min_cpus_per_task" ]]; then
    export min_cpus_per_task=1
fi

##check if interactive job
if  \$interactive ; then

    echo "test_driver.sh: interactive run - setting JOBID to \$\$"
    export JOBID=\$\$
    if [ \$0 = "test_driver.sh" ]; then
	initdir="."
    else
	initdir=\${0%/*}
    fi
fi

##establish script dir and cam_root
if [ -f \${initdir}/test_driver.sh ]; then
    export CAM_SCRIPTDIR=\`cd \${initdir}; pwd \`
    export CAM_ROOT=\`cd \${CAM_SCRIPTDIR}/../../../.. ; pwd \`
else
    if [ -n "\${CAM_ROOT}" ] && [ -f \${CAM_ROOT}/components/cam/test/system/test_driver.sh ]; then
	export CAM_SCRIPTDIR=\`cd \${CAM_ROOT}/components/cam/test/system; pwd \`
    else
	echo "ERROR: unable to determine script directory "
	echo "       if initiating batch job from directory other than the one containing test_driver.sh, "
	echo "       you must set the environment variable CAM_ROOT to the full path of directory containing "
        echo "       <components>. "
	exit 3
    fi
fi

##output files
cam_log=\${initdir}/td.\${JOBID}.log
if [ -f \$cam_log ]; then
    rm \$cam_log
fi
cam_status=\${initdir}/td.\${JOBID}.status
if [ -f \$cam_status ]; then
    rm \$cam_status
fi

##setup test work directory
if [ -z "\$CAM_TESTDIR" ]; then
    export CAM_TESTDIR=\${MACH_WORKSPACE}/\$LOGNAME/test-driver.\${JOBID}
    if [ -d \$CAM_TESTDIR ]; then
        rm -rf \$CAM_TESTDIR
    fi
fi
if [ ! -d \$CAM_TESTDIR ]; then
    mkdir -p \$CAM_TESTDIR
    if [ \$? -ne 0 ]; then
	echo "ERROR: unable to create work directory \$CAM_TESTDIR"
	exit 4
    fi
fi

##set our own environment vars
export CSMDATA=\${dataroot}/inputdata
export MPI_TYPE_MAX=100000

##process other env vars possibly coming in
if [ -z "\$CAM_RETAIN_FILES" ]; then
    export CAM_RETAIN_FILES=FALSE
fi
if [ -n "\${CAM_INPUT_TESTS}" ]; then
    input_file=\$CAM_INPUT_TESTS
else
    input_file=\${CAM_SCRIPTDIR}/\${input_file}
fi

if [ ! -f \${input_file} ]; then
    echo "ERROR: unable to locate input file \${input_file}"
    exit 5
fi

if \$interactive ; then
    echo "reading tests from \${input_file}"
else
    echo "reading tests from \${input_file}" >> \${cam_log}
fi

num_tests=\`wc -w < \${input_file}\`
echo "STATUS OF CAM TESTING UNDER JOB \${JOBID};  scheduled to run \$num_tests tests from:" >> \${cam_status}
echo "\$input_file" >> \${cam_status}
echo "" >> \${cam_status}
if [ ! \$interactive ]; then
    echo "see \${cam_log} for more detailed output" >> \${cam_status}
fi
echo "" >> \${cam_status}

test_list=""
while read input_line; do
    test_list="\${test_list}\${input_line} "
done < \${input_file}

##initialize flags, counter
skipped_tests="NO"
pending_tests="NO"
count=0

##loop through the tests of input file
for test_id in \${test_list}; do
    count=\`expr \$count + 1\`
    while [ \${#count} -lt 3 ]; do
        count="0\${count}"
    done

    master_line=\`grep \$test_id \${CAM_SCRIPTDIR}/input_tests_master\`
    status_out=""
    for arg in \${master_line}; do
        status_out="\${status_out}\${arg} "
    done


    test_cmd=\${status_out#* }

    status_out="\${count} \${status_out}"

    if  \$interactive ; then
        echo ""
        echo "************************************************************"
        echo "\${status_out}"
        echo "************************************************************"
    else
        echo "" >> \${cam_log}
        echo "************************************************************"\
            >> \${cam_log}
        echo "\$status_out" >> \${cam_log}
        echo "************************************************************"\
            >> \${cam_log}
    fi

#    if [ \${#status_out} -gt 64 ]; then
#        status_out=\`echo "\${status_out}" | cut -c1-70\`
#    fi
    while [ \${#status_out} -lt 95 ]; do
        status_out="\${status_out}."
    done

    echo \$echo_arg "\$status_out\c" >> \${cam_status}



    if \$interactive ; then
        \${CAM_SCRIPTDIR}/\${test_cmd} \$CAM_RBOPTIONS
        rc=\$?
    else
        \${CAM_SCRIPTDIR}/\${test_cmd} \$CAM_RBOPTIONS >> \${cam_log} 2>&1
        rc=\$?
    fi
    if [ \$rc -eq 0 ]; then
        if [ \${CAM_RBOPTIONS} = "build_only" ]; then
          echo "BUILT at \$(date)" >> \${cam_status}
        else
          echo "PASS at \$(date)" >> \${cam_status}
        fi
    elif [ \$rc -eq 255 ]; then
        echo "SKIPPED* at \$(date)" >> \${cam_status}
        skipped_tests="YES"
    elif [ \$rc -eq 254 ]; then
        echo "PENDING** at \$(date)" >> \${cam_status}
        pending_tests="YES"
    else
        echo "FAIL! rc= \$rc at \$(date)" >> \${cam_status}
	if \$interactive ; then
	    if [ "\$CAM_SOFF" != "FALSE" ]; then
		echo "stopping on first failure"
		echo "stopping on first failure" >> \${cam_status}
		exit 6
	    fi
	else
	    if [ "\$CAM_SOFF" = "TRUE" ]; then
		echo "stopping on first failure" >> \${cam_status}
		echo "stopping on first failure" >> \${cam_log}
		exit 6
	    fi
	fi
    fi
done


echo "end of input" >> \${cam_status}
if  \$interactive ; then
    echo "end of input"
else
    echo "end of input" >> \${cam_log}
fi

if [ \$skipped_tests = "YES" ]; then
    echo "*  please verify that any skipped tests are not required of your cam commit" >> \${cam_status}
fi
if [ \$pending_tests = "YES" ]; then
    echo "** tests that are pending must be checked manually for a successful completion" >> \${cam_status}
    if  ! \$interactive ; then
	echo "   see the test's output in \${cam_log} " >> \${cam_status}
	echo "   for the location of test results" >> \${cam_status}
    fi
fi

EOF
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ writing to batch script ^^^^^^^^^^^^^^^^^^^

cb_flag=`expr $cb_flag + 1`
done
cat >> ${submit_script} << EOF
exit 0
EOF
if [ ${submit_script_cb} ]; then
cat >> ${submit_script_cb} << EOF
echo "bsub < ${submit_script}" >> \${cam_log}
bsub < ${submit_script} >> \${cam_log} 2>&1
exit 0
EOF
fi


for file in ${submit_script} ${submit_script_cb}
do
  chmod a+x $file
done


if $interactive ; then
    if [ ${submit_script_cb} ]; then
      ${submit_script_cb}
    else
      ${submit_script}
    fi
    exit 0
fi

if ! $force ; then 
    echo ""
    echo "**********************"
    echo "$submit_script has been created and will be submitted to the batch queue..."
    echo "(ret) to continue, (a) to abort"
    read ans
    case $ans in
	[aA]* ) 
	echo "aborting...type ./test_driver.sh -h for help message"
	exit 0
	;;
    esac
fi

echo "submitting..."
case $hostname in
    ##firefly
    ff* )  bsub < ${submit_script};;

    ##yellowstone
    ye* | ys* | ca*)  bsub < ${submit_script_cb};;

    ##hobart  
    hob* | h[[:digit:]]* )  qsub ${submit_script};;

    ##leehill
    le* ) ${submit_script};;

    ##edison
    ed* ) qsub ${submit_script};;

    ##jaguarpf/titan
    ja* | yo* ) qsub ${submit_script};;

    ##hopper
    hop* | ni* ) qsub ${submit_script};;

    ##lynx
    ly* ) qsub ${submit_script};;

esac
exit 0
