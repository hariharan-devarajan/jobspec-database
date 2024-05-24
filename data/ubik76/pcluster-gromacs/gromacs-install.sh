#!/bin/bash

# Install Spack

export SPACK_ROOT=/shared/spack
mkdir -p $SPACK_ROOT
cd $SPACK_ROOT/..
git clone -c feature.manyFiles=true https://github.com/spack/spack 
echo "export SPACK_ROOT=$SPACK_ROOT" >> $HOME/.bashrc
echo "source \$SPACK_ROOT/share/spack/setup-env.sh" >> $HOME/.bashrc
source $HOME/.bashrc

# Add AWS Binary cache to spack

spack mirror add binary_mirror https://binaries.spack.io/develop
spack buildcache keys --install --trust

# Setup Spack

spack compiler find

cat << EOF > $SPACK_ROOT/etc/spack/packages.yaml
packages:
    intel-mpi:
        externals:
        - spec: intel-mpi@2020.4.0
          prefix: /opt/intel/mpi/2021.4.0/
        buildable: False
    libfabric:
        variants: fabrics=efa,tcp,udp,sockets,verbs,shm,mrail,rxd,rxm
        externals:
        - spec: libfabric@1.13.2 fabrics=efa,tcp,udp,sockets,verbs,shm,mrail,rxd,rxm
          prefix: /opt/amazon/efa
        buildable: False
    openmpi:
        variants: fabrics=ofi +legacylaunchers schedulers=slurm ^libfabric
        externals:
        - spec: openmpi@4.1.1 %gcc@7.3.1
          prefix: /opt/amazon/openmpi
    pmix:
        externals:
          - spec: pmix@3.2.3 ~pmi_backwards_compatibility
            prefix: /opt/pmix
    slurm:
        variants: +pmix sysconfdir=/opt/slurm/etc
        externals:
        - spec: slurm@21.08.8-2 +pmix sysconfdir=/opt/slurm/etc
          prefix: /opt/slurm
        buildable: False
EOF

# Improve usability of spack for Tcl Modules
spack config --scope site add "modules:default:tcl:all:autoload: direct"
spack config --scope site add "modules:default:tcl:verbose: True"
spack config --scope site add "modules:default:tcl:hash_length: 6"
spack config --scope site add "modules:default:tcl:projections:all: '{name}/{version}-{compiler.name}-{compiler.version}'"
spack config --scope site add "modules:default:tcl:all:conflict: ['{name}']"
spack config --scope site add "modules:default:tcl:all:suffixes:^cuda: cuda"
spack config --scope site add "modules:default:tcl:all:environment:set:{name}_ROOT: '{prefix}'"
spack config --scope site add "modules:default:tcl:openmpi:environment:set:SLURM_MPI_TYPE: 'pmix'"
spack config --scope site add "modules:default:tcl:openmpi:environment:set:OMPI_MCA_btl_tcp_if_exclude: 'lo,docker0,virbr0'"
spack config --scope site add "modules:default:tcl:intel-oneapi-mpi:environment:set:SLURM_MPI_TYPE: 'pmi2'"
spack config --scope site add "modules:default:tcl:mpich:environment:set:SLURM_MPI_TYPE: 'pmi2'"

# Install Gromacs from Binary Cache
spack install gromacs

# Download input files

sudo yum install -y bsdtar
mkdir -p /shared/input/gromacs
wget -qO- https://www.mpinat.mpg.de/benchRIB | bsdtar xf - -C /shared/input/gromacs
wget -qO- https://www.mpinat.mpg.de/benchMEM | bsdtar xf - -C /shared/input/gromacs

# Create submission file and submit job
cat << EOF > /shared/input/gromacs/submit.sh
#!/bin/bash
#SBATCH --job-name=gromacs-run
#SBATCH --ntasks=36
#SBATCH --output=%x_%j.out
#SBATCH --partition=queue0
#SBATCH --constraint=c5n.18xlarge

spack load gromacs

mkdir -p /shared/jobs/101
cd /shared/jobs/101
mpirun -n 18 gmx_mpi mdrun -ntomp 2 -s /shared/input/gromacs/benchRIB.tpr -resethway

EOF


# Install VMD
cd $HOME
wget https://www.ks.uiuc.edu/Research/vmd/vmd-1.9.3/files/final/vmd-1.9.3.bin.LINUXAMD64-CUDA8-OptiX4-OSPRay111p1.opengl.tar.gz
tar xf vmd-1.9.3.bin.LINUXAMD64-CUDA8-OptiX4-OSPRay111p1.opengl.tar.gz
cd vmd-1.9.3/
./configure
cd src/
sudo make install


