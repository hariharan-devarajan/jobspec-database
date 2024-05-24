#!/bin/bash
#SBATCH -n 1
#SBATCH -t 11-00:00:00
#SBATCH --mem 49512
#SBATCH -o %A_Sepidermidis_res.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nbrazeau@med.unc.edu

snakemake -s run_CGEtools.snake.py --cluster "sbatch -n1 -t 1-00:00:00 --mem 49152 -o Cluster_%A_job.out" -j 8
