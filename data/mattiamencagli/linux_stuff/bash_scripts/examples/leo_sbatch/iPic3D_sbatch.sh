#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition boost_usr_prod
#######SBATCH --qos boost_qos_dbg
#SBATCH --time 00:20:00            # format: HH:MM:SS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4       # n tasks out of 32
#######SBATCH --cpus-per-task=1    # 4 virtual cpu per task. So I use n physical core
#SBATCH --gres=gpu:4               # 1 gpus per node out of 4
#SBATCH --job-name=iPic3D
#SBATCH --mail-type=NONE
#SBATCH --mail-user=m.mencagli@cineca.it

## load needed modules on the node
source ${HOME}/modules_files/iPic3D_mod

## print GPUs node information
#nvidia-smi

DIR="${HOME}/programming/iPic3D/"
cd $DIR

branch="coils"
#branch="EB"

scorep_check="n"

if [ ${branch} = "master" ]; then
	if [ ${scorep_check} = "n" ]; then
		## normal compilation
		mkdir -p build/
		cd build/
		mkdir -p data/
		cmake .. > cmakeout_prova 2>&1
		time make -j -B > makeout_prova 2>&1
		mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM.inp
	elif [ ${scorep_check} = "y" ]; then
		## compile iPic with scorep
		source ${HOME}/my_programs/ON_spack_scorep
		mkdir -p build_scorep/
		cd build_scorep/
		mkdir -p data/
		SCOREP_WRAPPER=off cmake .. -DCMAKE_CXX_COMPILER=scorep-g++ > cmakeout_prova 2>&1
		#export SCOREP_ENABLE_TRACING=true
		export SCOREP_TOTAL_MEMORY=4G
		time make -j -B > makeout_prova 2>&1
		scan mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM.inp
		square -s scorep_*
	fi
elif [ ${branch} = "coils" ]; then
	if [ ${scorep_check} = "n" ]; then
		## normal compilation
		mkdir -p build_coils/
		cd build_coils/
		mkdir -p data/
		cmake .. -DIPIC_PETSC_SOLVER=ON > cmakeout_prova 2>&1
		time make -j -B > makeout_prova 2>&1
		mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM_forCoils.inp
	elif [ ${scorep_check} = "y" ]; then
		## compile iPic with scorep
		source ${HOME}/my_programs/ON_spack_scorep
		mkdir -p build_coils_scorep/
		cd build_coils_scorep/
		mkdir -p data/
		SCOREP_WRAPPER=off cmake .. -DCMAKE_CXX_COMPILER=scorep-g++ -DIPIC_PETSC_SOLVER=ON > cmakeout_prova 2>&1
		#export SCOREP_ENABLE_TRACING=true
		export SCOREP_TOTAL_MEMORY=4G
		time make -j -B > makeout_prova 2>&1
		scan mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM_forCoils.inp
		square -s scorep_*
	fi	
elif [ ${branch} = "EB" ]; then
	if [ ${scorep_check} = "n" ]; then
		## normal compilation
		mkdir -p build_EB/
		cd build_EB/
		mkdir -p data/
		cmake .. > cmakeout_prova 2>&1
		time make -j -B > makeout_prova 2>&1
		mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM.inp
	elif [ ${scorep_check} = "y" ]; then
		## compile iPic with scorep
		source ${HOME}/my_programs/ON_spack_scorep
		mkdir -p build_EB_scorep/
		cd build_EB_scorep/
		mkdir -p data/
		SCOREP_WRAPPER=off cmake .. -DCMAKE_CXX_COMPILER=scorep-g++ > cmakeout_prova 2>&1
		#export SCOREP_ENABLE_TRACING=true
		export SCOREP_TOTAL_MEMORY=4G
		time make -j -B > makeout_prova 2>&1
		scan mpirun -np 4 ./iPic3D ${DIR}/inputfiles/GEM.inp
		square -s scorep_*
fi

#nsys profile --trace=cuda,mpi,nvtx,openacc --force-overwrite true -o report_MHD_highorder_BLAST_B1_gpu4_leo mpirun -np 4 ./pluto
#nsys profile --force-overwrite true -o report_MHD_BLAST_B3_304cube_leo_o2 ./pluto -maxsteps 10 -xres 304
#ncu --target-processes all -k regex:Roe_Solver -f -o NCUreport_MHD_BLAST_B3_304cube_leo ./pluto -maxsteps 2 -xres 304

