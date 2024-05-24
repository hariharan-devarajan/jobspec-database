#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 15
#SBATCH --mem-per-cpu=2048
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END
#SBATCH --partition long

module load openmpi/4.0.1
#export LD_LIBRARY_PATH=/opt/lammps-7Aug19_nnp_plumed/lib/nnp/lib


mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_114_16_79_112_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_149_16_112_148_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_149_16_79_148_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_150_16_112_148_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_150_16_79_148_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_80_16_112_79_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_80_16_148_79_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_81_16_112_79_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_17_81_16_148_79_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_41_178_49_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_41_178_85_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_41_178_94_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_42_178_49_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_42_178_85_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_42_178_94_40_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_50_178_40_49_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_50_178_85_49_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_50_178_94_49_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_51_178_40_49_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_180_51_178_85_49_0-2_0-3_500_40000_index1.lmp
rm core*
