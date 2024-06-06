#!/bin/sh

#######
#
#	Description
#	
#######

beginTime1=`perl -e 'print time;'`
date1=`date`
#PBSnodeFile=`cat $PBS_NODEFILE`
#PBSnodeFile=`cat $PBS_NODEFILE`
echo ""
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
#echo "~~ ${PBS_JOBID} ~ ${date1} ~ ${PBS_O_HOST} ~ ${PBS_O_WORKDIR} ~ ${PBSnodeFile} ~~"
echo "~~ ${PBS_JOBID} ~ ${date1} ~ ${PBS_O_HOST} ~ ${PBS_O_WORKDIR} ~ ${PBS_NODEFILE} ~~"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "Working variables:" $mainDir1 $TrueFake $Pheno $Subset $perm1
echo ""

#mainDir1 Ex: /data/userdata/pg/michaelt/Data/ALL_MAPPING/Pools/MM5/PostMerge/mapping_pool_merged/GATK/EPACTSFiles/Permutations/FakePheno/Pheno1/Exonic.Nonsynonymous/
#TrueFake Ex: FakePheno
#Pheno Ex: Pheno1
#Subset Ex: Exonic.Nonsynonymous
#perm1 Ex: 1

#Purposely leaving -rnd_seed as the default timestamp since technically it shouldn't matter with no permutations being done

java -Xmx512m -cp /home/michaelt/Software/GSEA2-2.1.0/gsea2-2.1.0.jar xtools.gsea.GseaPreranked -gmx /home/michaelt/Data/GSEA/c2.all.v4.0.symbols.gmt -rnk ${mainDir1}/AllPools.Vs2.QCed.preGATK.QCed.samplesMerged.rmdup.BQSR.calmd.AllPoolsMerged.ChrAll.GATK.RRs.UG.VQSR.SNP.PASSts99_9.wAA.Bi.DropOffTarg_1kb.geno95.hwe1e4.wHM3.justWhite.QCed.DropIBD.EPACTS.${Subset}.${Pheno}.maf1.skat0.perm${perm1}.GSEA.rnk -nperm 0 -set_min 5 -collapse false -make_sets false -include_only_symbols true -gui false -plot_top_x 0 -zip_report false -rpt_label DataProcessing.Pipeline.pt6.All.ptg.GSEA.PermutationPrerankAnalyses.vs1.EPACTS.${Subset}.${Pheno}.maf1.skat0.perm${perm1}.GSEA -out ${mainDir1}/GSEA  

#java -Xmx512m -cp /home/michaelt/Software/GSEA2-2.1.0/gsea2-2.1.0.jar xtools.gsea.GseaPreranked -gmx /home/michaelt/Data/GSEA/c2.all.v4.0.symbols.gmt -rnk /data/userdata/pg/michaelt/Data/ALL_MAPPING/Pools/MM5/PostMerge/mapping_pool_merged/GATK/EPACTSFiles/Permutations/TruePheno/Pheno1/Exonic.Nonsynonymous/DataProcessing.Pipeline.Utility.CollectPValsPerGeneAcrossMultRuns.vs1.TruePheno.Pheno1.Exonic.Nonsynonymous.output.noNAs.Routput.rnk -nperm 0 -rnd_seed 432142 -set_min 5 -collapse false -make_sets false -include_only_symbols true -gui false -plot_top_x 0 -zip_report false -rpt_label DataProcessing.Pipeline.Utility.CollectPValsPerGeneAcrossMultRuns.vs1.TruePheno.Pheno1.Exonic.Nonsynonymous.output.noNAs.Routput -out /data/userdata/pg/michaelt/Data/ALL_MAPPING/Pools/MM5/PostMerge/mapping_pool_merged/GATK/EPACTSFiles/Permutations/TruePheno/Pheno1/GSEA

endTime1=`perl -e 'print time;'`
#timeDiff1=$((($endTime1-$beginTime1)/60/60))
timeDiff1=$(($endTime1-$beginTime1))
echo ""
echo "Script run time: $timeDiff1 ($endTime1 - $beginTime1)"
echo ""
