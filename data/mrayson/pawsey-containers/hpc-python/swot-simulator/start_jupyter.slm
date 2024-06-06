#!/bin/bash -l
# Allocate slurm resources, edit as necessary
#SBATCH --account=pawsey0106
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --job-name=swotsim_notebook
#SBATCH --output=swotsim-%j.out
#SBATCH --export=NONE
 
# Set our working directory
# This is the directory we'll mount to /home/jovyan in the container
# Should be in a writable path with some space, like /scratch
notebook_dir=$1 #"${MYSCRATCH}/"
 

#Set these to have singularity bind data locations
#export SINGULARITY_BINDPATH=/software:/software,/scratch:/scratch,/run:/run,$HOME:$HOME 
export SINGULARITY_BINDPATH=$MYSOFTWARE:$MYSOFTWARE,$MYSCRATCH:$MYSCRATCH,/run:/run,$HOME:$HOME 


#This is needed to setup conda in the container correctly
export SINGULARITYENV_PREPEND_PATH=/srv/conda/envs/notebook/bin:/srv/conda/condabin:/srv/conda/bin
export SINGULARITYENV_XDG_DATA_HOME=$MYSCRATCH/.local

# OpenMP settings
export SINGULARITYENV_OMP_NUM_THREADS=8  #To define the number of threads
export SINGULARITYENV_OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible)
export SINGULARITYENV_OMP_PLACES=cores     #To bind threads to cores


# End user-specfified environment variables
###

# Set the image and tag we want to use
image="docker://mrayson/swot_simulator:latest"
 
# You should not need to edit the lines below
 
# Prepare the working directory
mkdir -p ${notebook_dir}
cd ${notebook_dir}
 
# Get the image filename
imagename=${image##*/}
imagename=${imagename/:/_}.sif
 
# Get the hostname of the setonix node
# We'll set up an SSH tunnel to connect to the Juypter notebook server
host=$(hostname)
 

# Load Singularity
module load singularity/3.8.6
 
# Pull our image in a folder
singularity pull $imagename $image
echo $imagename $image
 
### Pangeo method for running notebooks
# Create trap to kill notebook when user is done
kill_server() {
    if [[ $JNPID != -1 ]]; then
        echo -en "\nKilling Jupyter Notebook Server with PID=$JNPID ... "
        kill $JNPID
        echo "done"
        exit 0
    else
        exit 1
    fi
}

let DASK_PORT=8787
let LOCALHOST_PORT=8888

JNHOST=$(hostname)
#JNIP=$(hostname -i)

LOGFILE=$MYSCRATCH/pangeo_jupyter_log.$(date +%Y%m%dT%H%M%S)


echo "Logging jupyter notebook session to $LOGFILE"

#jupyter notebook $@ --no-browser --ip=$JNHOST >& $LOGFILE &
### Launch our container
### and mount our working directory to /home/jovyan in the container
### and bind the run time directory to our home directory
srun --export=ALL -m block:block:block singularity exec \
  -B ${notebook_dir}:/home/joyvan \
  -B ${notebook_dir}:${HOME} \
  ${imagename} \
  jupyter notebook $@ \
  --no-browser \
  --port=${LOCALHOST_PORT} --ip=$JNHOST \
  --notebook-dir=${notebook_dir} >& $LOGFILE &
  #-B ${notebook_dir}:/home/joyvan \
  #-B ${notebook_dir}:${HOME} \
  #-B/tmp:/tmp \
  # Attempt at linking slurm...
  #-B $(mktemp -d):/run/user \
  #-B /etc/slurm,/usr/lib64/liblua5.3.so.5,/usr/lib64/liblua.so.5.3,/usr/lib64/libmunge.so.2,/usr/lib64/slurm \
  #-B /usr/bin/sbatch,/usr/bin/scancel,/usr/bin/squeue,/var/run/munge,/run/munge  \
  #-B /usr/lib64/libreadline.so,/usr/lib64/libhistory.so,/usr/lib64/libtinfo.so,/usr/lib64/libjson-c.so.3 \

JNPID=$!

echo -en "\nStarting jupyter notebook server, please wait ... "

ELAPSED=0
ADDRESS=

while [[ $ADDRESS != *"${JNHOST}"* ]]; do
    sleep 1
    ELAPSED=$(($ELAPSED+1))
    #ADDRESS=$(grep -e '^\[.*\]\s*http://.*:.*/\?token=.*' $LOGFILE | head -n 1 | awk -F'//' '{print $NF}')
    ADDRESS=$(grep -e '\s*http://.*:.*/\?token=.*' $LOGFILE | head -n 1 | awk -F'//' '{print $NF}')

    #echo ADDRESS=$ADDRESS

    if [[ $ELAPSED -gt 60 ]]; then
        echo -e "something went wrong\n---"
        cat $LOGFILE
        echo "---"
        kill_server
    fi
done

echo -e "done\n---\n"

HOST=$(echo $ADDRESS | awk -F':' ' { print $1 } ')
PORT=$(echo $ADDRESS | awk -F':' ' { print $2 } ' | awk -F'/' ' { print $1 } ')
TOKEN=$(echo $ADDRESS | awk -F'=' ' { print $NF } ')

cat << EOF
Run the following command on your desktop or laptop:
    ssh -f -N -l $USER -L ${LOCALHOST_PORT}:${JNHOST}:$PORT -L $DASK_PORT:${JNHOST}:$DASK_PORT setonix.pawsey.org.au
Log in with your username/password or SSH keys (there will be no prompt).
Then open a browser and go to http://localhost:${LOCALHOST_PORT}. The Jupyter web
interface will ask you for a token. Use the following:
    $TOKEN
Note that anyone to whom you give the token can access (and modify/delete)
files in your PAWSEY spaces, regardless of the file permissions you
have set. SHARE TOKENS RARELY AND WISELY!
To stop the server, press Ctrl-C.
EOF

# Wait for user kill command
sleep inf



