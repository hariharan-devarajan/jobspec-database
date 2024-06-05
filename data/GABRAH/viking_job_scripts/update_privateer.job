#!/bin/bash
#SBATCH --job-name=install_privateer        # Job name
#SBATCH --mail-type=BEGIN,END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=hb1115@york.ac.uk      # Where to send mail
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --mem=4gb                        # Job memory request
#SBATCH --time=00:20:00                  # Time limit hrs:min:sec
#SBATCH --output=install_privateer_%j.log        # Standard output and error log
#SBATCH --account=chem-structglyco-2019

module purge

module load devel/CMake/3.16.4-GCCcore-9.3.0
module load lang/Python/3.8.2-GCCcore-9.3.0

cmake --version || "CMake is not found!. Install CMake then re-run this script " || exit 3
gcc --version  || "GCC is not found!. Install GCC then re-run this script " || exit 3

GCC="$(which gcc)"
GPLUSPLUS="$(which g++)"
echo "GCC: $GCC"
echo "GCC: $GPLUSPLUS"
GFORTRAN="$(which gfortran)"
echo "GFORTRAN: $GFORTRAN"
Threads="$(nproc --all)"

export CC=$GCC
export CXX=$GPLUSPLUS
export FC=$GFORTRAN

mainDir=/users/hb1115/scratch/PDB_anal
cd $mainDir
cd $mainDir/privateer
privateerDir=$PWD

echo Current working directory is `pwd`
echo Running job on host:
echo -e '\t'`hostname` at `date`
echo
echo Privateer directory is located at $privateerDir


git checkout privateerpython
git pull
git submodule update --init --recursive
source $privateerDir/pvtpython/bin/activate
source $privateerDir/ccp4.envsetup-sh
pip install -r requirements.txt

cd $privateerDir

python3 $privateerDir/setup.py install


echo Job completed at `date`