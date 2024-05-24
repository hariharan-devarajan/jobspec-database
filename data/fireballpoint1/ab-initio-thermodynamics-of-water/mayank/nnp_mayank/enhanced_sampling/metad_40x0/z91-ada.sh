#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 15
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

module load u18/openmpi/4.1.2
#export LD_LIBRARY_PATH=/opt/lammps-7Aug19_nnp_plumed/lib/nnp/lib


mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_96_171_94_178_169_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_96_179_94_109_178_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_96_179_94_169_178_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_96_180_94_109_178_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_96_180_94_169_178_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_131_97_133_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_131_97_25_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_132_97_133_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_132_97_25_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_134_97_130_133_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_134_97_25_133_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_135_97_130_133_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_135_97_25_133_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_26_97_130_25_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_26_97_133_25_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_27_97_130_25_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_98_27_97_133_25_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_99_131_97_133_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_99_131_97_25_130_0-2_0-3_500_40000_index1.lmp
mpirun -np 15 /opt/n2p2/bin/lmp_mpi < nvt_share_H_heated_cooled_away_close_metad_99_132_97_133_130_0-2_0-3_500_40000_index1.lmp
rm core*
