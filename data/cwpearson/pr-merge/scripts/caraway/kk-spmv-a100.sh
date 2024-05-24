#!/bin/bash
#SBATCH -N 1
#SBATCH -p MI100
#SBATCH --time=01:00:00
#SBATCH -o kk-smpv-a100.o%j
#SBATCH -e kk-smpv-a100.e%j

shopt -s extglob

ROOT=/home/cwpears/repos/pr-merge

source $HOME/spack-caraway/spack/share/spack/setup-env.sh
spack load cuda
module load cmake

date

echo "reals_med"
echo "path,rows,nnz,us,GFLOPS,GB/s"
for m in /home/projects/cwpears/suitesparse/reals_med/*.mtx; do
#   "$ROOT"/build-caraway/kokkos-kernels/perf_test/sparse/sparse_spmv -t kk_kernels -f $m
  out=`"$ROOT"/build-caraway-a100/kokkos-kernels/perf_test/sparse/sparse_spmv -t kk-kernels -f $m`
  out=$(echo "$out" | head -n3 | tail -n1)
  nnz=$(echo $out | tr -s ' ' | cut -d' ' -f1)
  nr=$(echo $out | tr -s ' ' | cut -d' ' -f2)
  avgBw=$(echo "$out" | tr -s ' ' | cut -d' ' -f6)
  avgGf=$(echo "$out" | tr -s ' ' | cut -d' ' -f11)
  avgMs=$(echo "$out" | tr -s ' ' | cut -d' ' -f16)
  avgUs=$(echo $avgMs*1000 | bc)
  echo $m,$nr,$nnz,$avgUs,$avgGf,$avgBw
done

date