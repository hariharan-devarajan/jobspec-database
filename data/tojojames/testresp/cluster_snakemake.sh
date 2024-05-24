A Snakefile along with a Slurm cluster configuration that uses a Slurm array for job parallelization. This Snakefile will run multiple tasks in parallel using Snakemake and Slurm's `sbatch`:

**Snakefile**:

```python
# Snakefile

# Define your rules here, similar to the previous example.
# For demonstration purposes, let's assume we have a rule called "task" that takes an input and generates an output.
# Replace this with your actual rules.

rule task:
    input:
        "input/{sample}.txt"
    output:
        "output/{sample}.processed.txt"
    conda:
        "envs/task.yaml"
    script:
        "scripts/process.sh {input} {output}"

# Cluster configuration
cluster_configfile: "cluster.yaml"

# Specify cluster-wide conda and Docker settings
conda: "envs/cluster.yaml"
container_img: "your-docker-image:latest"
```

**cluster.yaml**:

```yaml
__default__:
  jobscript: cluster_jobscript.sh
  account: your_account              # Your Slurm account name
  partition: your_partition          # The partition/queue to use
  time: '24:00:00'                   # Maximum runtime for jobs (HH:MM:SS)
  cpus-per-task: 4                   # Number of CPU cores per task
  mem: 8G                            # Memory per job (e.g., 8 gigabytes)
  job-name: snakemake_array_job
  array: 1-10                        # Specify the range of array tasks (e.g., 1 to 10)
```

**cluster_jobscript.sh**:

```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --partition=your_partition
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=snakemake_array_job
#SBATCH --array=1-10    # Specify the range of array tasks (e.g., 1 to 10)

# Activate Conda environment (if needed)
# conda activate your_conda_environment

# Go to the working directory
cd $SLURM_SUBMIT_DIR

# Run Snakemake for the current array task
snakemake --cores $SLURM_CPUS_PER_TASK --use-conda --use-singularity --keep-going --jobname snakemake_task.{%} --rerun-incomplete --unlock --resources mem_mb=8000 --keep-target-files

# Optionally, log the job's completion status to a file
if [ $? -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully" >> task_completion.log
else
    echo "Task $SLURM_ARRAY_TASK_ID failed" >> task_completion.log
fi
```

In this example:

- The Snakefile contains a rule called "task," which you can replace with your actual rules. This rule assumes that it processes input files in parallel and generates corresponding output files.

- The `cluster.yaml` file is configured to use Slurm array jobs with a range of tasks (1 to 10 in this example).

- The `cluster_jobscript.sh` script is the job script that is submitted to Slurm for each task in the array. It activates the Conda environment (if needed), goes to the working directory, runs Snakemake for the current task, and logs the task's completion status.

Remember to replace placeholders like `your_account`, `your_partition`, `your_conda_environment`, and `your-docker-image:latest` with your actual settings. Additionally, customize the rules, scripts, and resources according to your specific workflow requirements.
