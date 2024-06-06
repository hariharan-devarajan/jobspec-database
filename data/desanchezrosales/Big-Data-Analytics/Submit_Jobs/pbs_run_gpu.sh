#PBS -N cnn_mnist_gpu
#PBS -l walltime=1:30:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=4761MB
#PBS -j oe

# uncomment if using qsub
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module load python/3.6-conda5.2
module load cuda/10.0.130
python -u cnn_intro.py >& cnn_intro_output_gpu.lg
