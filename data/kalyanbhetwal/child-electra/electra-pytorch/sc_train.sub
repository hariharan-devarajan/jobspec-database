#!/bin/bash

#SBATCH -J electra     # job name
#SBATCH -N 1 		  # number of nodes you want to run on	
#SBATCH --partition=adaptlab
#SBATCH -n 28
#SBATCH -t 12:00:00       # run time (hh:mm:ss) - 12.0 hours in this example.
#SBATCH --output=electra1.log
# Generally needed modules:
module load slurm
module load cuda10.2


# Execute the program

## Some examples:
# mpirun python3 script.py

source  "/home/kbhetwal/anaconda3/etc/profile.d/conda.sh"
conda activate torch
export PYTHONPATH=${PYTHONPATH}:/home/kbhetwal/child-electra/electra-pytorch


#python pretraining/openwebtext/pretrain.py --gpu -1


python examples/glue/run.py  --model_type electra --model_name_or_path google/electra-large-discriminator --data_dir data/glue_data/MRPC  --task_name MRPC --output_dir output/MRPC_electra_large

python examples/glue/run.py  --model_type electra  --model_name_or_path google/electra-large-discriminator --data_dir data/glue_data/CoLA  --task_name CoLA --output_dir output/cola_electra_large

python examples/glue/run.py   --model_type electra --model_name_or_path google/electra-base-discriminator --data_dir data/glue_data/WNLI  --task_name WNLI --output_dir output/wnli_electra_large



# Exit if mpirun errored:
status=$?
if [ $status -ne 0 ]; then
    exit $status
fi

# Do some post processing.
