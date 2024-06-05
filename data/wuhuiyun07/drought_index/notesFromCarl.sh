#!/bin/bash
#SBATCH --job-name=install_conda    ### Job Name
#SBATCH --output=install_conda.%j.out       ### File in which to store job output
#SBATCH --error=install_conda.%j.err        ### File in which to store job error messages
#SBATCH --qos=long         ### Quality of Service (like a queue in PBS)
#SBATCH --time=2-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Number of tasks to be launched per Node
#SBATCH --cpus-per-task=20      # Number of threads per task (OMP threads)
#SBATCH --partition=centos7      # use centos 7 operation system

#CentOS7 nodes are currently all being used. You probably have to wait for a while. Once you get logging in,
#If you haven’t set, set CONDA_ENVS_PATH to somewhere common in your group, for example,
export CONDA_ENVS_PATH=/lustre/project/taw/Wu_analysis/conda_envs
module load anaconda3/2020.07
unset PYTHONPATH
#If you want to create a new local environment, for example, wu-base,
#conda create -y -n wu-base python
#Install ‘conda’
#conda install -y -n wu-base -c defaults conda
#Activate the environment,
source $CONDA_ENVS_PATH/wu-base/etc/profile.d/conda.sh
conda activate wu-base
#Check the version of ‘conda’
conda --version
#It should be 23.7.3

conda install -c conda-forge mamba

mamba --version



idev --partition=centos7
#export SINGULARITY_CACHEDIR=<lustre singularity cache dir.>
module load singularity/3.9.0
source /lustre/project/singularity_images/setup_cypress.sh
#singularity pull docker://ghcr.io/chanzuckerberg/czid-workflows/czid-short-read-mngs-public:latest
singularity shell -s /bin/bash czid-short-read-mngs-public_latest.sif
#<now in shell>
#export PATH=/share/apps/anaconda/3/2020.07/bin:$PATH
export CONDA_ENVS_PATH=/lustre/project/taw/Wu_analysis/conda_envs
#<create env.>
#conda create -y -n sing-base python
#<Install conda>
#conda install -y -n sing-base -c defaults conda
conda list -n sing-base 
# <Only now at this point I'm getting an error...>
#ResolvePackageNotFound:
 # - python=3.1
#<...otherwise proceed...>
#<activate the env.>
unset PYTHONPATH          
source $CONDA_ENVS_PATH/sing-base/etc/profile.d/conda.sh
conda activate sing-base
# <at this point Huiyun gets the included error running "conda --version">


####run after Hideki's email 5:52 pm 8/28/23

idev74 ##on centos 7 node, 4 hours
module load anaconda3/2023.07 
module load singularity/3.9.0
source activate miniwdl
export MINIWDL_CFG=/share/apps/centos7/anaconda3/2023.07/envs/miniwdl/miniwdl.cfg
source /lustre/project/singularity_images/setup_cypress.sh
# Clone the czid-workflows repo:
# git clone https://github.com/chanzuckerberg/czid-workflows.git
cd czid-workflows
# Run the viral pipeline:
miniwdl run workflows/short-read-mngs/local_driver.wdl    \
docker_image_id=ghcr.io/chanzuckerberg/czid-workflows/czid-short-read-mngs-public     \
fastqs_0=workflows/short-read-mngs/test/norg_6__nacc_27__uniform_weight_per_organism__hiseq_reads__v6__R1.fastq.gz    \
fastqs_1=workflows/short-read-mngs/test/norg_6__nacc_27__uniform_weight_per_organism__hiseq_reads__v6__R2.fastq.gz     \
-i workflows/short-read-mngs/test/local_test_viral.yml --verbose

 
