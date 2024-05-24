#! /bin/bash

### This script is a modified copy of "raxml-mpi.quick_test_1k.slurm" from /scratch/dbrow208/galick_gun_working_dir/20220522_test_raxml/ -dbrow208 20220525

# ===== SLURM OPTIONS =====
#SBATCH --partition=Pisces
#SBATCH --job-name=raxml_900_ws_PW		# ALTER JOB NAME.
#SBATCH --nodes=1			# Attempts to prevent crashing by separating jobs, SHOULD ALWAYS BE 1.
#SBATCH --ntasks-per-node=6		# CHANGE PER BELOW NOTE. The number of distinct tree searches, without overwhelming available RAM or cores.
#SBATCH --mem=375gb			# Designates a Pisces Dual 18-Core Intel Xeon Gold 6154 CPU @ 3.00GHz (36 cores / node) with 388GB RAM total
#SBATCH --output=%x_%j.out
#SBATCH --time=14-00:00:00

### NOTE: for -ntasks-per-node ($SLURM_NTASKS_PER_NODE) and -N flag. -dbrow208 20220525
# RAxML is not fully parallel, so only distinct tree searches may be assigned to processors per the manual (pg. 2) https://cme.h-its.org/exelixis/resource/download/NewManual.pdf
# Designate the -ntasks-per-node (applied as -N below), based on estimated memory requirments from calculator utilizing taxa and patterns found at https://cme.h-its.org/exelixis/web/software/raxml/
# As an example, for 1000 taxa and 588000 patterns, the memory is estimated at 71GB. With 4 tasks set, each will be allowed 93.75GB RAM out of the allowed 375GB. That is more than enough.
# To achieve total number of desired replicates, THIS SCRIPT should be renamed and submitted according to the formula: # replicates desired / --ntasks-per-node = # of times to submit this script.

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Node List   : $SLURM_JOB_NODELIST"
echo "Num Tasks   : $SLURM_NTASKS total [$SLURM_NNODES nodes @ $SLURM_CPUS_ON_NODE CPUs/node]"
echo "======================================================"
echo ""

####################

##########
### REQUIRED SLURM OPTIONS
##########

### Executable name (uncomment one)
#RAXML_BIN="raxmlHPC-MPI"
RAXML_BIN="raxmlHPC-MPI-AVX2"
#RAXML_BIN="raxmlHPC-MPI-SSE3"

### Choose version (uncomment one)
RAXML_VER="8.2.12"
#RAXML_VER="8.2.4"
#RAXML_VER="7.4.2"


####################

##########
### MAJOR RAxML PROGRAM OPTIONS - change all three
##########

### Set program options for path to alignment file (-s == RAXML_SEQ), output prefix (-n == RAXML_OUT), and path to an optional starting tree (-t == STARTING_TREE).
RAXML_SEQ="/scratch/dbrow208/galick_gun_working_dir/subset_900/results_Roary_no_split/core_gene_alignment.aln"
#RAXML_OUT="900_ws_core_opt"
RAXML_OUT="900_ws_core_opt_PAIRWISE_DIST"
STARTING_TREE="/scratch/dbrow208/galick_gun_working_dir/subset_900/RAxML_bestTree.with_split"
UNROOTED_TREE="/scratch/dbrow208/galick_gun_working_dir/subset_900/RAxML_bestTree.900_ws_core_opt"
PW_REF_TREE="/scratch/dbrow208/galick_gun_working_dir/subset_900/RAxML_rootedTree.900_ws_core_opt.ROOTED"

####################

##########
### PRESET CHECKS AND TESTS - uncomment one to run
##########

### Checks if the alignment can be read by RAxML
#RAXML_OPTS="-f c -m GTRCAT"
### Execute VERY fast experimental tree search (only for testing)
#RAXML_OPTS="-f E -m GTRCAT -p 1234"
### Execute fast experimental tree search (only for testing)
#RAXML_OPTS="-f F -m GTRCAT -p 1234"
##########

####################

##########
### ANALYSIS EXAMPLES - uncomment one to run and/or alter the provided options.
##########

##########
### Additional RAxML options defined below
### **DO NOT ENTER** -s or -n or -N options. Those have already been defined and are called below.
##########

### True replicates (full) are -N while boostraps are -b. Do NOT use bootstraps. - db 20220525

### Feel free to change between -m options (GTRGAMMA and GTRCAT) based on your needs. Please see RAxML manual. - db 20220605

### Standard run with replicates (-N) from above
#RAXML_OPTS="-f a -m GTRGAMMA -p 1234 -x 1234"
### Final thorough optimization of ML tree after a standard run
#RAXML_OPTS="-f T -m GTRGAMMA -t $STARTING_TREE -p 1234"
### Rapid hill-climbing algorithm
#RAXML_OPTS="-f d -m GTRCAT -p 1234"
### Rapid hill-climbing with bootstraps
#RAXML_OPTS="-f D -m GTRCAT -p 1234"
### Rapid hill-climbing algorithm with user designated starting tree
#RAXML_OPTS="-f d -m GTRCAT -t $STARTING_TREE -p 1234"
### Simple tree rooting algorithm for unrooted trees by rooting at branch that best balances subtree lengths (sum over branches in the subtrees) ??? midpoint? - db 20220930
#RAXML_OPTS="-f I -m GTRGAMMA -t $UNROOTED_TREE -n ${RAXML_OUT}.ROOTED"
### Compute pair-wise ML distances. Can pass tree with '-t' , might only work for GAMMA-based models
RAXML_OPTS="-f x -p 1234 -m GTRGAMMA -t $PW_REF_TREE"

####################

##########
### DO NOT CHANGE BELOW
##########

### Call the submit script for Slurm and run RAxML
cd $SLURM_SUBMIT_DIR
module load raxml/${RAXML_VER}-mpi
srun --mpi=pmix_v3 $RAXML_BIN -s $RAXML_SEQ -n $RAXML_OUT -N $SLURM_NTASKS_PER_NODE $RAXML_OPTS

echo ""
echo "======================================================"
echo "End Time   : $(date)"
echo "======================================================"
