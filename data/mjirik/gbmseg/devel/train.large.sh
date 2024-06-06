#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb:scratch_local=10gb:cluster=adan -q gpu_long
#PBS -l walltime=27:00:00
#PBS -o /storage/plzen1/home/hachaf/projects/gbm/out.log
# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.
# add to qsub with:
# qsub train.sh
# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
DATADIR="/storage/plzen1/home/$LOGNAME/data/"
# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT
# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1
# spusteni aplikace - samotny vypocet
export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
module add cuda
activate tf_gpu
python /storage/plzen1/home/$LOGNAME/projects/gbm/train.py /storage/plzen1/home/hachaf/projects/gbm/settings.large.json > /storage/plzen1/home/$LOGNAME/projects/gbm/results.out
