#!/bin/bash
##
## Lines starting with #SBATCH are read by Slurm. Lines starting with ## are comments.
## All other lines are read by the shell.
##
## Basic parameters
##
#SBATCH --account=priority-jenniferlachowiec      # specify the account to use if using a priority partition
#SBATCH --partition=priority                # queue partition to run the job in
#SBATCH --cpus-per-task=120               # number of cores to allocate
#SBATCH --mem=100G                        # ammount of Memory allocated
#SBATCH --time=0-24:00:00               # maximum job run time in days-hours:minutes:secconds
#SBATCH --job-name=HiSat2              # job name
#SBATCH --output=example-%j.out         # standard output from job
#SBATCH --error=example-%j.err          # standard error from job
#SBATCH --mail-user=brodysturgis@gmail.com      # enter your email to recieve email notifications
#SBATCH --mail-type=ALL                 # specify conditions on which to recieve emails
##
## Optional parameters - remove one leading hashtag to enable
##
#SBATCH --nodes=1                      # number of nodes to allocate for multinode job
#SBATCH --ntasks-per-node=1            # number of descrete tasks to allocate for MPI job
#SBATCH --array=1-3                    # number of jobs in array for job array

## Run 'man sbatch' for more information on the options above.
## Below, enter commands needed to execute your workload
# Example for basic test
#date                                    # print out the date
#hostname -s                             # print a message from the compute node
#date                                    # print the date again


source ~/.bashrc
module load Anaconda3/2022.05
conda activate brody
for file_r1 in $(ls /home/group/jenniferlachowiec/2023/202302_csativa_flowers/data/raw_rnaseq/*.1.fastq.gz)
do
  file_r2=`echo $file_r1 | sed 's/.1.fastq.gz/.2.fastq.gz/'`
  samFile=`basename $file_r1 | sed 's/.1.fastq.gz/.sam/'`
  logFile=`basename $file_r1 | sed 's/.1.fastq.gz/.log/'`
  hisat2 -p 120 -x /home/group/jenniferlachowiec/2023/202302_csativa_flowers/data/reference/camelina_index -1 $file_r1 -2 $file_r2 -S /home/group/jenniferlachowiec/2023/202302_csativa_flowers/results/hisat2/$samFile --summary-file /home/group/jenniferlachowiec/2023/202302_csativa_flowers/results/hisat2/$logFile
done
