#!/usr/bin/bash
sbatch<<EOT
#!/usr/bin/bash
#SBATCH -J $1 #job_name
#SBATCH --array=$2-$3
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --output=result/$5/$1/%a/output.txt
#SBATCH --cpus-per-task=1
#$6

source ~/.bashrc
conda activate env39

srun python main3.py -c conf/$1.yaml -w $4 -o $5/$1/$4
sleep 60

EOT