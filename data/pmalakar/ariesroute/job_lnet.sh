#!/bin/bash
#COBALT -t 10
#COBALT -q default
#COBALT -A Performance

echo "COBALT_JOBID" $COBALT_JOBID
echo "COBALT_PARTNAME" $COBALT_PARTNAME
echo "COBALT_JOBSIZE" $COBALT_JOBSIZE

export MPICH_RANK_REORDER_DISPLAY=true

depth=1
rpn=32 #1

#FILE=tloc_165332.txt.ost10.lnetroutes
#FILE=tloc_165332.txt.ost9.lnetroutes
#FILE=tloc_165332.txt.ost10.iter1

#echo $FILE

#echo ${COBALT_PARTNAME} > location.txt
#python parsenodes.py theta.computenodes location.txt > nodetable

#this file will give you rank to node mapping, if required
#python parsejobnodes.py theta.computenodes location.txt > jobnodetable

python parsenodes.py theta.allnodes tloc_187190.txt > nodetable
#python parsejobnodes.py theta.allnodes tloc_165332.txt > jobnodetable

#change the file name to whatever, this contains information about Hops
#requires nodetable, and tau output file $FILE

for FILE in tloc_187190.txt.lnetroutes   #tloc_165332.txt.ost10.iter1 tloc_165332.txt.ost10.iter9 tloc_165332.txt.ost10.iter10
do

  OUTPUT=hopinfo_${FILE}_${COBALT_JOBID}

  #aprun --attrs mcdram=cache:numa=quad -n $((COBALT_JOBSIZE*rpn)) -N $rpn -d $depth -j 1 -cc depth ./multiroutes $FILE > $OUTPUT
  aprun -n $((COBALT_JOBSIZE*rpn)) -N $rpn -d $depth -j 1 ./multiroutes $FILE > $OUTPUT

done

status=$?
exit $status

