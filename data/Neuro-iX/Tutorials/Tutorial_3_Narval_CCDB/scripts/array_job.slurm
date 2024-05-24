#!/bin/bash
#SBATCH --job-name=<JOB NAME>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=<CPU NUMBER>
#SBATCH --mem=<RAM IN GIGA>G
#SBATCH --gpus-per-node=<GPU NUMBER>
#SBATCH --time=1:00:00
#SBATCH --account=<SUPERVISOR ACCOUNT>
#SBATCH --mail-user=<YOUR MAIL>
#SBATCH --mail-type=ALL
#SBATCH --array=<FROM>-<TO>

# Load Python module (assuming it's installed on your cluster)
module load python
module load cuda

# Create and activate a virtual environment
python -m venv monai
source monai/bin/activate

# Install system-level dependencies (adjust as needed)
pip install numpy --no-index
pip install scipy --no-index
pip install scikit-learn --no-index

# Install PyTorch and MONAI (adjust versions as needed)
pip install torch torchvision torchaudio --no-index
pip install monai --no-index
pip install nibabel --no-index
pip install pandas --no-index
pip install tqdm --no-index



# Additional dependencies can be installed here if needed
python brain_vae.py $SLURM_ARRAY_TASK_ID
# Deactivate the virtual environment
deactivate