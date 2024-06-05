#!/bin/bash 
### ### #SBATCH --partition neutron-star
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --mem 60G
#SBATCH --time 240:00:00
#SBATCH -w helios-cn003
##### ###SBATCH --array=13-17
#### #SBATCH --array=1-650%200
#### #SBATCH --exclude=helios-cn[016-020]
### ####SBATCH --output=/home/idayan/LOGALL/AVERAGE-PROCESS.log
###  #### SBATCH --output=/zfs/helios/filer0/idayan/Cal60-20200812/calibration.log
###  #### ((SBATCH --output=/home/idayan/CALwith60Mhz/calibration.log))

echo "cwd is"
#echo $pwd
echo $(pwd)

echo "database starting"
/hddstore/idayan/pgsql/data/bin/pg_ctl -D /hddstore/idayan/pgsql/data/c -l logfile start # node 3
#source /home/idayan/env/bin/activate
#source /home/idayan/new_env/bin/activate
source ~/envforTRAP/bin/activate
#JOBNAME = $2
echo "jobname is:"
echo $1
echo "starting processing"

trap-manage.py run $1  -m "[[293.73,21.90], [293.73, 19.40],[293.73, 23.40],[291.43, 21.90]]"
#trap-manage.py run ALEXSGR  -m "[[293.73,21.90], [293.73, 19.40],[293.73, 23.40],[291.43, 21.90]"
#trap-manage.py run  $1  -m "[[148.56,7.66], [148.56, 10.16],[148.56, 5.16],[151.06, 7.66],[146.06, 7.66]]"
#trap-manage.py run  _2Averaged20K101102JOB  -m "[[148.56,7.66], [148.56, 10.16],[148.56, 5.16],[151.06, 7.66],[146.06, 7.66]]"

#python /home/idayan/dataframe2/AVERAGE/average-frq-corr.py
#python /home/idayan/dataframe2/AVERAGE/averager-meancorr.py
#python /home/idayan/dataframe2/AVERAGE/average.py


