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


mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_35_103_31_34_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_36_103_136_34_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_36_103_13_34_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_36_103_1_34_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_36_103_31_34_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_3_103_136_1_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_3_103_13_1_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_3_103_31_1_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_104_3_103_34_1_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_137_103_13_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_137_103_1_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_137_103_31_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_137_103_34_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_138_103_13_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_138_103_1_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_138_103_31_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_138_103_34_136_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_14_103_136_13_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_14_103_1_13_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /global/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_105_14_103_31_13_0-2_0-3_500_40000_index1.lmp
rm core*
