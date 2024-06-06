#!/bin/bash
#BSUB -P CSC143
#BSUB -W 60
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J adios2
#BSUB -nnodes 8
##BSUB -alloc_flags "smt4"

source $PWD/module-to-load-summit.sh

set -x
JOBID=$LSB_JOBID
[ -z $JOBSIZE ] && JOBSIZE=$(((LSB_DJOB_NUMPROC-1)/42))

#METHOD=BPFile
#METHOD=SST
METHOD=InSituMPI
[ $# -gt 0 ] && METHOD=$1
[ -z $XR ] && XR=6
[ -z $FR ] && FR=1
[ -z $LOUT ] && LOUT=9
[ -z $PM ] && PM=1
[ -z $NPTL ] && NPTL=100K
[ -z $TPR ] && TPR=1
[ -z $CPR ] && CPR=6
[ -z $RPN ] && RPN=7
[ -z $OMP ] && OMP=6
[ -z $GPU ] && GPU=0
[ -z $NSTEP ] && NSTEP=20
[ -z $NTIME ] && NTIME=1

PPN=$((RPN*TPR))
MAX_OMP=$((42/PPN))
if [ $OMP -gt $MAX_OMP ]; then
    echo "!! WARNING !! OMP is larget than MAX_OMP ($OMP, $MAX_OMP)"
fi

if [ $GPU -eq 1 ]; then
    _GPU=-gpu
else
    _GPU=
fi

export MPICH_CPUMASK_DISPLAY=1
export MPICH_RANK_REORDER_DISPLAY=1

NNODES=$JOBSIZE
NP=$((NNODES*PPN))

XGC_NP=$((NP*XR/(XR+FR)))
FTT_NP=$((NP*FR/(XR+FR)))
if [ $FTT_NP -gt 0 ] && [ $FTT_NNODE -eq 0 ]; then
    FTT_NNODE=1
fi
echo "PPN,OMP,MAX_OMP= $PPN $OMP $MAX_OMP"
echo "XGC_NP,FTT_NP= $XGC_NP $FTT_NP"

SUFFIX=$JOBID-N$NNODES-PPN$PPN-OMP$OMP-$XGC_NP-$FTT_NP-$METHOD
WDIR=w-$SUFFIX
echo $WDIR
mkdir $WDIR
cd $WDIR

function comm_setup() {
    ENGINE=$1
    NSUB=$2
    mkdir -p timing
    mkdir -p restart_dir
    rm -f *.unlock
    if [ $ENGINE == "BPFile" ]; then
    cat <<EOF > adios2cfg.xml
<?xml version="1.0"?>
<adios-config>
    <io name="f0"> 
        <engine type="BPFile">
            <parameter key="Profile" value="Off"/>
            <parameter key="SubStreams" value="$NSUB"/>
            <!--
            <parameter key="QueueLimit" value="1"/>
            <parameter key="ControlTransport" value="sockets"/>
            -->
        </engine>
    </io>
    <io name="f3d">
        <engine type="BPFile">
            <parameter key="Profile" value="Off"/>
            <parameter key="SubStreams" value="$NSUB"/>
            <parameter key="CollectiveMetadata" value="Off"/>
            <!--
            -->
        </engine>
    </io>
    <io name="diagnosis.f0.mesh">
        <engine type="BPFile">
            <parameter key="SubStreams" value="1"/>
        </engine>
    </io>
</adios-config>
EOF
    else
    cat <<EOF > adios2cfg.xml
<?xml version="1.0"?>
<adios-config>
    <io name="f0"> 
        <engine type="$ENGINE">
            <parameter key="Profile" value="Off"/>
            <parameter key="verbose" value="3"/>
            <parameter key="OpenTimeoutSecs" value="600"/>
            <!--
            <parameter key="verbose" value="3"/>
            <parameter key="QueueLimit" value="1"/>
            <parameter key="ControlTransport" value="sockets"/>
            <parameter key="SubStreams" value="$NSUB"/>
            -->
        </engine>
    </io>
    <io name="f3d">
        <engine type="BPFile">
            <parameter key="Profile" value="Off"/>
            <parameter key="SubStreams" value="$NSUB"/>
            <parameter key="CollectiveMetadata" value="Off"/>
            <!--
            -->
        </engine>
    </io>
    <io name="diagnosis.f0.mesh">
        <engine type="BPFile">
            <parameter key="SubStreams" value="1"/>
        </engine>
    </io>
</adios-config>
EOF
    fi

    if [ $ENGINE == "BPFile" ]; then
        sed -i "s/adios_stage_f0=.*./adios_stage_f0=.false./g" adios_in
    else
        sed -i "s/adios_stage_f0=.*./adios_stage_f0=.true./g" adios_in
    fi
    if [ $NNODES -ge 64 ]; then
        sed -i "s/sml_nphi_total=.*./sml_nphi_total=$((NNODES/32))/g" input
        sed -i "s/sml_grid_nrho=.*./sml_grid_nrho=6/g" input
    else
        sed -i "s/sml_nphi_total=.*./sml_nphi_total=2/g" input
        sed -i "s/sml_grid_nrho=.*./sml_grid_nrho=2/g" input
    fi

    #if [ $FTT_NP -gt 0 ] && [ $FTT_NP -le $((32*12)) ]; then
    #    sed -i "s/sml_nphi_total=.*./sml_nphi_total=2/g" adios_in
    #else
    #    sed -i "s/sml_nphi_total=.*./sml_nphi_total=32/g" adios_in
    #fi
    sed -i "s/sml_mstep=.*./sml_mstep=$NSTEP/g" input
    sed -i "s/adios_f0_ntimes=.*./adios_f0_ntimes=$NTIME/g" adios_in

    sed -i "s/sml_electron_on=.*./sml_electron_on=.true./g" input
    sed -i "s/sml_ncycle_half=.*./sml_ncycle_half=30/g" input
}

BASEDIR=/gpfs/alpine/scratch/swithana/csc143/xgc-f/exp-xgc-ftt-demo
EXE_DIR=/gpfs/alpine/scratch/swithana/csc143/xgc-f/XGC-Devel-xgc1-f0-coupling/xgc_build

cp $BASEDIR/setup2/*.sh .
cp -r $BASEDIR/setup2/xgc_work .
cp -r $BASEDIR/setup2/ftt_work .
cp -r $BASEDIR/setup2/xgc_base .
ln -snf $BASEDIR/setup2/XGC-1_inputs .
ln -snf xgc_work coupling
ln -snf $BASEDIR/*.py .

pushd xgc_work
ln -snf $EXE_DIR/xgc-es* .
comm_setup $METHOD $NNODES
NPTL=$(numfmt --from=si $NPTL)

## Assume MAX OMP:7
NPTL_NUM=$((NPTL*NNODES*6*7)) # Assume MAX OMP:7
NPTL_NUM_PER_CPU=$((NPTL_NUM/XGC_NP/OMP))
echo NPTL_NUM_PER_CPU=$NPTL_NUM_PER_CPU
sed -i "s/ptl_num=.*./ptl_num=$NPTL_NUM_PER_CPU/g" input
sed -i "s/diag_f3d_period=.*./diag_f3d_period=1000/g" input
popd
pushd ftt_work
ln -snf $EXE_DIR/xgc-f0* .
comm_setup $METHOD $NNODES
sed -i "s/diag_f3d_period=.*./diag_f3d_period=1/g" input
rm -rf restart_dir
ln -snf ../coupling/restart_dir
popd
pushd xgc_base
ln -snf $EXE_DIR/xgc-es* .
comm_setup BPFile $NNODES
sed -i "s/adios_stage_f0=.*./adios_stage_f0=.false./g" adios_in
sed -i "s/diag_f3d_period=.*./diag_f3d_period=1/g" input
sed -i "s/diag_f0_period=.*./diag_f0_period=1000/g" input
sed -i "s/ptl_num=.*./ptl_num=$NPTL_NUM_PER_CPU/g" input
popd

echo "START: " `date`
cat $LSB_DJOB_HOSTFILE | uniq | tail -n +2 > NODEFILE.$JOBID
cat NODEFILE.$JOBID | head -n $JOBSIZE > NODEFILE
which python
env > env-$JOBID.info

XGC_EXE="xgc-es"
FTT_EXE="xgc-f0"

date
JSRUN="jsrun -X 0 --erf_output erf_file.out" #-e prepended
export OMP_NUM_THREADS=$OMP
if [ $FTT_NP -eq 0 ]; then
    python reorder-hyper-summit.py --smt=4 --nnodes=$NNODES  \
        0,$CPR,0,3:"./run-base$_GPU.sh":-g 21,$CPR,0,3:"./run-base$_GPU.sh":-g
    touch coupling/ready.f0.unlock
    ## job placement
    $JSRUN --erf_input=erf_file 2>&1 | tee run.log
else
    python reorder-hyper-summit.py --smt=4 --nnodes=$NNODES  \
        0,6,0,3:"./run-xgc$_GPU.sh":-g 21,6,0,3:"./run-xgc$_GPU.sh":-g "18,3 39,3":"./run-ftt.sh"
    ## job placement
    $JSRUN --erf_input=erf_file 2>&1 | tee run.log
fi
date

echo "DONE: " `date`
