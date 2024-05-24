#!/bin/bash                       
#SBATCH --time=02:00:00
#SBATCH --account=
#SBATCH --mem 128000
#SBATCH --job-name=
#SBATCH --output=
#SBATCH --mail-user=

# path to Rosetta and input file folder
ROSETTA3="/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/MPI/intel2016.4/openmpi2.1/rosetta/3.10"
ROSETTA3_DB="/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/MPI/intel2016.4/openmpi2.1/rosetta/3.10/database"
module load nixpkgs/16.09  gcc/7.3.0  openmpi/3.1.2 rosetta/3.10

# Define output file paths. -out:path:all flag to define rosetta output path. 
relaxed_files=
backrub_files=
mutated_files=

# Clean PDB file with ligand using a python script in Rosetta/tools, which removed all lines that do not start with ATOM. 
# Ligand atoms - HETATM fields are added back to this cleaned pdb afterwards (called 5icu_HETATM.pdb)
# Input file name: 5icu.pdb
python $ROSETTA3/tools/protein_tools/scripts/clean_pdb.py 5icu.pdb ignorechain

# Basic energy scoring based on crystallographic energy functions. Add tag to ignore phosphate and other unrecognized components. 
$ROSETTA3/bin/score_jd2.mpi.linuxiccrelease -in:file:s 5icu_HETATM.pdb -ignore_unrecognized_res -out:path:all $output_files -out:suffix _original

# Run Relax protocol, output 10 potential PDBs and select the lowest. 
$ROSETTA3/bin/relax.mpi.linuxiccrelease -in:file:s 5icu_HETATM.pdb -out:path:all $relaxed_files -nstruct 10

# Backrub (not used in the process since all output PDBs have higher energy scores than the selected relaxed structure
# $ROSETTA3/bin/backrub.mpi.linuxiccrelease -in:file:s 5icu_HETATM_relaxed.pdb -out:suffix _backrubbed -nstruct 10 -in:file:fullatom -backrub:ntrials 10000 -out:path:all $backrub_files

# Point mutation protocol. @pmut.flags specifies the flags, and mutants.txt specifies the mutants list. 
# Mutants are listed in the format of: <chain_id> <old amino acid> <residue number> <new amino acid> 
$ROSETTA3/bin/pmut_scan_parallel.mpi.linuxiccrelease -in:file:s 5icu_HETATM_relaxed.pdb @pmut.flags

# Generate scores for all mutated proteins. List the protein pdb file names in score_pdb.txt 
$ROSETTA3/bin/score_jd2.mpi.linuxiccrelease -in:file:l score_pdb.txt -ignore_unrecognized_res -out:path:all $mutated_files