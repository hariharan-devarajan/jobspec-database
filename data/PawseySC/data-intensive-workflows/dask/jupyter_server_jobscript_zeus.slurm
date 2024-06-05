#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --export=NONE

module load python/3.6.3
module load jupyter/1.0.0
module load numpy pandas h5py matplotlib dask scikit-image scikit-learn


# get tunneling info
export XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
cluster="zeus"
port=8888


# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -N -f -L ${port}:${node}:${port} ${user}@${cluster}.pawsey.org.au

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# Run Jupyter
jupyter-notebook --no-browser --port=${port} --ip=${node}

