declare -a arr=(
"VBFHToMuMu"
"DYJetsToLL"
#"Data2017"
"ggH"
"ttTosemileptonic"
"ttTo2l2v"
"ZH"
"WminusH"
"WplusH"
"ttH"
"WZTo1L1Nu2Q"
"ZZ"
"WZTo3LNu"
"WZTo2L2Q"
"WWTo2L2Nu"
"WWToLNuQQ"
"WWW_4F"
"WWZ_4F"
"ttJets_DiLept"
"TTZToLLNuNu"
"TTWJetsToLNu"
)
## now loop through the above array
for i in "${arr[@]}"
do
#split -d -l 10 -a 4 ${1}.txt ${1}.splitted_list
 count=0
 while read p; do
   #echo "$p"
   count=$((count+1))
   prefix=${i}_${count}
   echo "$p" > ${prefix}_input_list.txt
   input=${prefix}_input_list.txt
   data=data
   path=/afs/cern.ch/work/i/idutta/public/CMSSW_9_4_9/src/HmmAnalyzer/HmmAna_ntuples/Histogram_RootFiles/
   opt=F
   echo $input $data $path $opt
   bsub -R "pool>100000" -q 8nh run.sh $input $data $path $prefix $opt
 done < ${i}.txt
 #input=${1}runList_data.txt
 #data=data
 #path=/eos/cms/store/user/nlu/Hmm/ntuple
 #opt=T
 #bsub -R "pool>100000" -q 1nh run.sh $input $data $path $prefix $opt
done
