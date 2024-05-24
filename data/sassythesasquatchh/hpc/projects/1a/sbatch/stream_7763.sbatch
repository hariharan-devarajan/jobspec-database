#!/bin/bash

PROJ_DIR="$HOME/projects/1a"
BIN_DIR="$PROJ_DIR/bin"
SRC_DIR="$PROJ_DIR/src"
LOGS_DIR="$PROJ_DIR/logs"

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --output="stream_7763.out"
#SBATCH --open-mode=truncate
#SBATCH --constraint=EPYC_7763

module load gcc
cd "$SRC_DIR"
gcc -O3 -march=native -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=16000000 \
	-DNTIMES=20 stream.c -o "$BIN_DIR/stream_c.exe"
"$BIN_DIR/stream_c.exe"

