#!/bin/bash

#Ekstra Aktivite 1: Argümanları İle Çalışan Bir Bash Scripti Oluşturmak
tRNAscan-SE -o $1 $2


# output alma
# tRNAscan-SE -o G_intestinalis.trna G_intestinalis.fasta

#Bu scripti aşağıdaki gibi çalıştırabiliriz:
#termin alde çalışıtr #bash run_trnascan_args.sh G_intestinalis.trna1 G_intestinalis.fasta

# Ekstra Aktivite 2: SLURM için Bir Sbatch Oluşturmak
#Eğer bir HPC (High Performance Computing) kümesi kullanıyorsanız ve bu küme SLURM iş
#zamanlamayı kullanıyorsa, `sbatch` dosyaları ile işlerinizi zamanlayabilirsiniz.

#SBATCH --job-name=trna_scan
#SBATCH --output=trna_scan_output.txt

#module load tRNAscan-SE
#tRNAscan-SE -o G_intestinalis.trna2 G_intestinalis.fasta
#Bu script, 'trna_scan' isminde bir iş oluşturur, çıktıyı 'trna_scan_output.txt' dosyasına yazar
# ve tRNAscan-SE'yi belirli bir genom dosyasında çalıştırır.


#Ekstra Aktivite 3: Snakemake Python Scripti Oluşturmak
#Snakemake, Python tabanlı bir iş akışı yönetim sistemidir. İşte basit bir Snakemake iş akışı:

rule trna_scan:
  input:
    "G_intestinalis.fasta"
  output:
    "G_intestinalis.trna2"
  shell:
    "tRNAscan-SE -o {output} {input}"