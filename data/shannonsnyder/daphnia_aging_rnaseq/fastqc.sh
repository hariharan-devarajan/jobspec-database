#!/bin/bash                                                                                                            
#SBATCH -A nereus                ### Account                                                                           
#SBATCH --partition=short          ### Partition                                                                       
#SBATCH --job-name=fastqc         ### Job Name                                                                         
#SBATCH --time=3:00:00        ### WallTime                                                                             
#SBATCH --nodes=1              ### Number of Nodes                                                                     
#SBATCH --ntasks=1             ### Number of tasks per array job                                                       
##SBATCH --array=0-3           ### Array index                                                                         

#SBATCH --mail-type=END              ### Mail events (NONE, BEGIN, END, FA$                                            
#SBATCH --mail-user=ssnyder3@uoregon.edu  ### Where to send mail                                                       
#SBATCH --cpus-per-task=8            ### Number of CPU cores per task                                                  
#SBATCH --requeue 

module load fastqc/0.11.5
module load easybuild
module load icc/2017.1.132-GCC-6.3.0-2.27
module load impi/2017.1.132
module load ifort/2017.1.132-GCC-6.3.0-2.27                                                                                          
module load eb-hide/1                                                                                        
module load eb-hide/1
module load MultiQC/1.3-Python-3.6.1
mkdir fastqc_out
mkdir multiqc_out
OUTDIR=fastqc_out

#input fastq dir                                                                                                       
FASTQDIR=/home/ssnyder3/nereus/aging_rnaseq/raw_fastq


# generate fastqc files                                                                                                
/usr/bin/time fastqc -o $OUTDIR -t 6 $FASTQDIR/*.fastq.gz

# generate multiQC report to summarize                                                                                 
/usr/bin/time multiqc $OUTDIR -o multiqc_out
