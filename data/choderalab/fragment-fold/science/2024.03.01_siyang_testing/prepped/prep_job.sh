#!/bin/bash
#BSUB -J PREP
#BSUB -n 1
#BSUB -R rusage[mem=50]
#BSUB -q gpuqueue
#BSUB -gpu num=1
#BSUB -W 10:00
#BSUB -o log_files/prep_orig_docked_ligands.out
#BSUB -e log_files/prep_orig_docked_ligands.stderr
source ~/.bashrc
cd /home/pengs/fold_zika
conda activate asapdiscovery

asap-prep protein-prep --target ZIKV-NS2B-NS3pro --structure-dir orig_bind --output-dir orig_prepped --align zikv_ns2b3.pdb --ref-chain B --active-site-chain B --use-dask

echo done
