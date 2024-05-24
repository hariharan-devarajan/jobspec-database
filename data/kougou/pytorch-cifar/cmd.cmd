#!/bin/bash
# parallel job using 16 processors. and runs for 4 hours (max)
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH -t 02:10:00
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=dsu@princeton.edu
#SBATCH --mem=20000
# Load openmpi environment
#module load anaconda
#export PATH=$PATH:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.181-3.b13.el7_5.x86_64/jre/lib/amd64/libjava.so:/home/dsu/anaconda2/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin:/opt/dell/srvadmin/bin:/home/dsu/.local/bin:/home/dsu/bin
export PATH=$PATH:/usr/lib/jvm/java-1.8.0-openjdk
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.181-3.b13.el7_5.x86_64/
export LD_LIBRARY_PATH=/usr/lib/jvm/jre/lib/amd64:$LD_LIBRARY_PATH

python main.py