#!/bin/bash
#BSUB -n 1 
#BSUB -o ejecutor_%J.out
#BSUB -e ejecutor_%J.err
#BSUB -J ejecutor
#BSUB -R"span[ptile=16]"
#BSUB -W 48:00

python get_cardinalities_3D_arrays.py top5_combos.all pdb.list.BM5
#/apps/GREASY/2.1.2.1/bin/greasy ROTSPIN.txt
#apps/GREASY/2.1.2.1/bin/greasy ordenes_bm4_zdock.txt
#python get_just_ligand_1KKL.py;
#python get_just_ligand_1N2C.py;
#python get_just_ligand_1Y64.py;
#python get_just_ligand_1XU1.py;
#python get_just_ligand_1F51.py;
