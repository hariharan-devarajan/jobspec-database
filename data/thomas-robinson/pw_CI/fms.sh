#!/bin/sh -x
#SBATCH --job-name=FMS_container
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --time=00:15:00               # Time limit hrs:min:sec
#SBATCH --output=FMS_container_%j.log   # Standard output and error log

buildDir=/lustre/build_2021.2
#Install singularity
sudo yum install -y singularity

#clone FMS
cd /lustre
git clone https://github.com/NOAA-GFDL/FMS.git

# Set up build
mkdir ${buildDir} && cd ${buildDir}
autoreconf -i ../FMS/configure.ac

#Build FMS
singularity exec -B /lustre,/contrib /contrib/intel2021.2_netcdfc4.7.4_ubuntu.sif autoreconf -i ../FMS/configure.ac
singularity exec -B /lustre,/contrib /contrib/intel2021.2_netcdfc4.7.4_ubuntu.sif /lustre/FMS/configure --prefix=${buildDir} FC=mpiifort CC="mpiicc -no-multibyte-chars" CPPFLAGS="-L/opt/netcdf-fortran/lib -lnetcdff -I/opt/netcdf-fortran/include -I/opt/netcdf-fortran/include -I/opt/netcdf-c/include -I/opt/hdf5/include -I/include -L/opt/netcdf-c/lib -lnetcdf" LIBS="-L/opt/netcdf-fortran/lib -lnetcdff -L/opt/netcdf-c/lib -lnetcdf" 

singularity exec -B /lustre,/contrib /contrib/intel2021.2_netcdfc4.7.4_ubuntu.sif /contrib/make_fms_container.sh
