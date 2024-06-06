#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ku_00004 -A ku_00004
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N assoctest
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e {$numeric_jobid}vcf_again.err
#PBS -o {$numeric_jobid}vcf_again.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes, request 196 cores from 7 nodes
#PBS -l nodes=1:ppn=40:thinnode
### Requesting time - 720 hours
#PBS -l walltime=04:00:00
 
### Here follows the user commands:
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
# NPROCS will be set to 196, not sure if it used here for anything.
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS nodes
  
module load openmpi/gcc/64/1.10.2  tools htslib/1.13 angsd/0.935

export OMP_NUM_THREADS=6
# Using 192 cores for MPI threads leaving 4 cores for overhead, '--mca btl_tcp_if_include ib0' forces InfiniBand interconnect for improved latency
#mpirun -np 40 $mdrun -s gmx5_double.tpr -plumed plumed2_path_re.dat -deffnm md-DTU -dlb yes -cpi md-DTU -append --mca btl_tcp_if_include ib0

#angsd -bam /home/projects/ku_00004/data/Kusel/contig_length_KU/bam_list.txt -out angsd_all_kusel_ten_node -GL 1 -doGlf 2 -doMajorMinor 1 -doCounts 1 -doMaf 1 -SNP_pval 1e-6 -doGeno 5 -baq 1 -minQ 20 -doPost 2 -postCutoff 0.5 -geno_minDepth 2 -only_proper_pairs 1 -ref BATG-0.5-CLCbioSSPACE.fa -anc BATG-0.5-CLCbioSSPACE.fa
#angsd -doAsso 6 -yQuant percentage_dead.txt    -cov test1.cov -nThreads OMP_NUM_THREADS   -nind 10 -fai ref.fai   -doMaf 1 -vcf-gl beagle_kusel_vcf.vcf.gz  -out Vcf_gp
angsd -doAsso 6 -yQuant /home/projects/ku_00004/data/bams/Match_bam_2_health.txt -Pvalue 1   -cov full_pca_2_cols.cov -nThreads 6   -fai Genome_bat/ref.fai   -doMaf 4 -beagle split_1_multi/cat_assoc_files/new_file_cat.txt > file 2>&1
#angsd -doAsso 6 -yQuant percentage_dead.txt    -cov test1.cov -nThreads OMP_NUM_THREADS    -doMaf 4 -vcf-gp beagle_kusel_vcf.vcf.gz -out Vcf_gp
#angsd -doAsso 6 -yQuant percentage_dead.txt     -doMaf 1 -vcf-gl angsd_vcf_test.bcf  -out Vcf_gp
