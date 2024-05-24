#!/bin/bash
# Primarily meant for use on the Rocky cluster at NIMBioS
if [[ ! -d moss_phylo ]] ; then 
echo Making functional term transformers test environment [ moss_phylo ]
echo Can install environment modules using easybuild
module load MAFFT/7.505-GCC-11.3.0-with-extensions FastTree/2.1.11-GCCcore-11.3.0 BLAST+/2.13.0-gompi-2022a MCL/22.282-GCCcore-11.3.0 phyx/1.3-foss-2022a Cython/0.29.33-GCCcore-11.3.0 Python/2.7.18-GCCcore-11.3.0-bare
echo Python should be from Python/2.7
python --version
virtualenv moss_phylo


source moss_phylo/bin/activate
# install required pacakages
# fastBio requires specific old versions...
# pip3 install fastBio==0.1.7
pip install pysqlite networkx clint
# check packages
pip freeze | cut -f 1 -d ' ' | column
# accelerate==0.16.0
# sentencepiece==0.1.97
# torch==1.12.1

deactivate # leave virtual environent
# reset loaded modules to defaults
module purge
module restore

fi

# Also, note this super useful slurm script from ACCRE cluster
# lifted from https://www.vanderbilt.edu/accre/documentation/python/
cat <<EOF | tee moss_phylo_stub.srun
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus_per_task=4
#SBATCH --output=moss_phylo_%j_slurm.out


module load MAFFT/7.505-GCC-11.3.0-with-extensions FastTree/2.1.11-GCCcore-11.3.0 BLAST+/2.13.0-gompi-2022a MCL/22.282-GCCcore-11.3.0 phyx/1.3-foss-2022a Cython/0.29.33-GCCcore-11.3.0 Python/2.7.18-GCCcore-11.3.0-bare

source ${PWD}/moss_phylo/bin/activate
EOF
cat <<"EOF" | tee -a moss_phylo_stub.srun
python --version

EOF
