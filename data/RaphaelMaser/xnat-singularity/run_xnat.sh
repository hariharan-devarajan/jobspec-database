#!/bin/sh
#SBATCH -J XNAT
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7                # Cores assigned to each tasks
#SBATCH --time=0-24:00:00
#SBATCH -p batch

module load tools/Singularity

if ! test -f "xnat-singularity.sif"; then
    echo "Singularity container needs to be build"
    singularity build --force --fakeroot xnat-singularity.sif xnat.def
    echo "Singularity container was built"
fi

mkdir -p xnat/plugins
mkdir -p xnat/data/home/logs
mkdir -p xnat/data/archive 
mkdir -p xnat/data/build
mkdir -p xnat/data/cache
mkdir -p xnat/data/configs
mkdir -p postgres/data
mkdir -p postgres/run
mkdir -p postgres/config

singularity run -f -e --no-home --writable \
-B postgres/data:/var/lib/postgresql/13/main \
-B postgres/config:/etc/postgresql/13/main \
-B xnat/plugins:/data/xnat/home/plugins \
-B xnat/data/home/logs:/data/xnat/home/logs \
-B xnat/data/archive:/data/xnat/archive \
-B xnat/data/build:/data/xnat/build \
-B xnat/data/cache:/data/xnat/cache \
-B xnat/data/configs:/data/xnat/home/configs \
xnat-singularity.sif \
xnat