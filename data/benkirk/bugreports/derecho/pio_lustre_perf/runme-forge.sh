#!/bin/bash
#PBS  -l select=16:ncpus=128:mpiprocs=128:ompthreads=1
#PBS  -q main
#PBS  -l walltime=00:30:00
#PBS  -m n
#PBS  -N pioperf
#PBS  -r n
#PBS  -j oe


inst_dir=$(pwd)/install/pio+forge

[ -d /glade/work/benkirk/bugreports/derecho/pio_lustre_perf/install/pio+forge ] && inst_dir="/glade/work/benkirk/bugreports/derecho/pio_lustre_perf/install/pio+forge"

#. ${inst_dir}/../../config_env.sh || exit 1
module reset >/dev/null 2>&1
module load gcc/12.2.0 >/dev/null 2>&1
module unload ncarcompilers hdf5 netcdf >/dev/null 2>&1
module load linaro-forge >/dev/null 2>&1
module list

exe=${inst_dir}/bin/pioperf

[ -x ${exe} ] || { echo "cannot locate executable!: ${exe}"; exit 1; }

ldd ${exe}

cat <<EOF > pioperf.nl
&pioperf
 decompfile='ROUNDROBIN'
 varsize=18560
 pio_typenames = 'pnetcdf', 'pnetcdf'
 rearrangers = 2
 nframes = 1
 nvars = 64
 niotasks = 16
 /

EOF


rm -f ./pioperf.*.nc

export MPICH_MPIIO_AGGREGATOR_PLACEMENT_DISPLAY=1
export MPICH_MPIIO_HINTS_DISPLAY=1
export MPICH_MPIIO_STATS=1
export MPICH_MPIIO_TIMERS=1
export MPICH_MPIIO_HINTS="pioperf*.nc:cray_cb_write_lock_mode=2:cray_cb_nodes_multiplier=12"
export FORGE_SAMPLER_INTERVAL=5 # (ms)

#mpiexec --label --line-buffer -n 2048 ${exe}
map --connect \
    --processes=2048 \
    --select-ranks=0,128,256,384,512,640,768,896,1024,1152,1280,1408,1536,1664,1792,1920 \
    --start \
    ${exe}
