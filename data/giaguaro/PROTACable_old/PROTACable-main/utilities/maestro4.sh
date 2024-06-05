#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=stage4

source ~/.bashrc
conda activate amber

m=$1
#for m in *pdb;
#do
mkdir ../stg4/${m%.*}_results
rm ../stg4/${m%.*}_results/*
cp $m ../stg4/${m%.*}_results/
cd ../stg4/${m%.*}_results/

$SCHRODINGER/utilities/structconvert -ipdb $m -omae pv_${m%.*}.mae

$SCHRODINGER/run pv_convert.py pv_${m%.*}.mae -mode split_pv -lig_last_mol

#pv_out_nhyd99_mae_amb_complex_poi_5j8o_28_docked_e3_cereblon_5j8o_28_docked_pp_model_xx01-out_pv.mae

$SCHRODINGER/prime_mmgbsa pv_${m%.*}-out_pv.mae -WAIT -NOJOBID -job_type SITE_OPT -target_flexibility -target_flexibility_cutoff 20 -out_type COMPLETE

#$SCHRODINGER/prime_mmgbsa pv_${m%.*}-out_pv.mae -HOST slurm-compute -NJOBS 1

# pv_out_nhyd99_mae_amb_6bn7_docked_p_model_xx01_cat-out-out.maegz
#ENERGY,REAL_MIN,SIDE_PRED,SIDE_COMBI,SITE_OPT,PGL
$SCHRODINGER/utilities/structconvert -imae pv_${m%.*}-out-out.maegz -opdb pv_${m%.*}-out-out.pdb

#../../process_4

mv *out-out* ../../process_4
#rm pv_${m%.*}-out-out.maegz pv_${m%.*}-out_pv.mae pv_${m%.*}.mae
echo "DONE MODELLING"

#done
