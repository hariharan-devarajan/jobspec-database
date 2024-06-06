#!/bin/bash
#SBATCH -J blob_array
#SBATCH -o blob_%A_%a.out
#SBATCH -e blob_%A_%a.err
#SBATCH -N 1 -n 10
#SBATCH -t 1:00:00
#SBATCH -p debug,mem-low,brief-low
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yelena.pacheco@usda.gov


#make blobtools plots of blobBlast.sh results
#need 1. assembly contig fasta 2.minimap coverage file 3. blobBlast.sh results

name1="N_clydesmithi"

#--- Starrt timer
t1=$(date +"%s")

echo "Starting $name1"
module load blobtools/4.3.6



# Generate blobdb
blobtools create --fasta N_clydesmithi_hifiasm_contigs.fa $name1

# add coverage
blobtools add --cov minimap.bam $name1

# add taxonomic info
blobtools add --hits N_clydesmithi.megablast_nt --taxrule bestsumorder --taxdump ./taxdump $name1

# plot
blobtools view --plot $name1 
# snail plot
blobtools view --plot --view snail  $name1


# plot colored at a different rank
blobtools view --plot --param catField=bestsumorder_genus $name1 --out $name1 # this outputs the genus level one into the individual folder.


# Make output tables at genus, order and phylum level
#blobtools view -i ${name1}_masurca.megablast_nt.blobDB.json -r genus -o genus
#blobtools view -i ${name1}_masurca.megablast_nt.blobDB.json -r order -o order
#blobtools view -i ${name1}_masurca.megablast_nt.blobDB.json -r phylum -o phylum

#Make blob plots at genus, order and phylum level
#blobtools blobplot -i ${name1}_masurca.megablast_nt.blobDB.json --format pdf -r genus -o genus
#blobtools blobplot -i ${name1}_masurca.megablast_nt.blobDB.json --format pdf -r order -o order
#blobtools blobplot -i ${name1}_masurca.megablast_nt.blobDB.json --format pdf -r phylum -o phylum

#---Complete job
t2=$(date +"%s")
diff=$(($t2-$t1))
echo "[---$SN---] ($(date)) $(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
echo "[---$SN---] ($(date)) $SN COMPLETE."