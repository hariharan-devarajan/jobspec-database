#!/bin/bash

#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=2:mem=8gb 
#PBS -l walltime=05:00:00
#PBS -M aponsero@email.arizona.edu
#PBS -m bea

module load perl
RUN="$WORKER_DIR/grind.pl"
HOST=`hostname`
LOG="$STDOUT_DIR/${HOST}.log"
ERRORLOG="$STDERR_DIR/${HOST}.log"

if [ ! -f "$LOG" ] ; then
    touch "$LOG"
fi
echo "Started `date`">>"$LOG"
echo "Host `hostname`">>"$LOG"

##################################################"
### grind
cd $DIR
for FILE in *.fna
do
    OUTFILE="$OUT/${SIZE_CONTIGS}_${STEP}_${FILE}"
    perl $RUN $FILE $SIZE_CONTIGS $OUTFILE $STEP
done

### merge and subsample
cd $OUT
cat *.fna > all_${SIZE_CONTIGS}_${STEP}.fasta

export NUM=$(grep -c '>' all_${SIZE_CONTIGS}_${STEP}.fasta)
echo "number found in file : $NUM" >> $LOG

# Check if number of remaining sequences are enough for the SPlit_size

if [ $SUBSET -gt $NUM ]; then
    echo "$NUM sequences are available. Please provide a $SUBSET inferior to $NUM in config file. Job aborted."
    exit 1
fi

# randomize the dataset and generates randomlists
module load python

export RUN="$WORKER_DIR/randomize.py"
export RESULT_DIR="$OUT/subsets_$SUBSET"
mkdir $RESULT_DIR
python3 $RUN -f all_${SIZE_CONTIGS}_${STEP}.fasta -o $RESULT_DIR -s $SUBSET -m $NUM -n $REPLICATE

### create bins
export BIN_DIR="$OUT/bins"
mkdir $BIN_DIR
cd $OUT

for (( c=1; c<=$REPLICATE; c++ ))
do
    REP_DIR="$BIN_DIR/rep_$c"
    mkdir $REP_DIR
    for FILE in *.fna
    do
        export NUM=$(grep -c '>' $FILE)
        export OUT_DIR="$REP_DIR/bins_$FILE"
        mkdir $OUT_DIR
        python3 $RUN -f $FILE -o $OUT_DIR -s $BIN_SIZE -m $NUM -n $NB_BIN_PER_GENOMES
    done
done

echo "Finished `date`">>"$LOG"
