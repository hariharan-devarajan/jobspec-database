#!/bin/bash
#SBATCH -J idr_bio
#SBATCH -o slurm/idr_bio-%j.out
#SBATCH -e slurm/idr_bio-%j.err
#SBATCH -N 1
#SBATCH -n 12
#SBATCH -t 1:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=END
#SBATCH --mail-user=tasawwar_rahman@brown.edu

module load python/3.6.6
module load gcc/6.2  
module load idr/2.0.2
# module load mpi/openmpi_3.1.6_gcc

mkdir bioreps/idr_bio
mkdir bioreps/idr_bio/logs
mkdir bioreps/idr_bio/images

cd bioreps/idr_bio

bio_replicates=(
#Non
ERR3975825_ERR3975848-ERR3975862_ERR3975871
#Neurons
ERR3975815_ERR3975819-ERR3975833_ERR3975864
#Myo
ERR3975860_ERR3975869-ERR3975846_ERR3975875
#Neurons 8-10
ERR3975830_ERR3975879-ERR3975820_ERR3975827)

for i in "${bio_replicates[@]}" 
do
echo idr bio ${i%%-*} and ${i#*-} 
idr --samples ../peaks/top100k_narrowPeak/${i%%-*}_top100k_peaks.narrowPeak ../peaks/top100k_narrowPeak/${i#*-}_top100k_peaks.narrowPeak \
    --input-file-type narrowPeak \
    --output-file ${i}.narrowPeak \
    --rank p.value \
    --only-merge-peaks \
    --log-output-file ${i}.idr.log

#--idr-threshold .05 \
#####No peaks passed this threshold, no peaks pass any threshold.
done

for i in "${bio_replicates[@]}" 
do
echo idr bio plot ${i%%-*} and ${i#*-} 
idr --samples ../peaks/top100k_narrowPeak/${i%%-*}_top100k_peaks.narrowPeak ../peaks/top100k_narrowPeak/${i#*-}_top100k_peaks.narrowPeak \
    --input-file-type narrowPeak \
    --output-file ${i}.temp \
    --rank p.value \
    --plot

mv ${i}.temp.png ${i}.png

done

mv *.log logs
mv *.png images
rm *.temp




#IGNORE:
# idr --samples ERR3975815_ERR3975819_top100k_peaks.narrowPeak  ERR3975833_ERR3975864_top100k_peaks.narrowPeak \
#     --input-file-type narrowPeak \
#     --rank p.value \
#     --output-file output \
#     --plot \
#     --log-output-file test.idr.log 


#    --dont-filter-peaks-below-noise-mean 
# ^^ USE THIS????

# idr --samples ERR3975815_ERR3975819_top100k_peaks.narrowPeak ERR3975833_ERR3975864_top100k_peaks.narrowPeak \
#     --input-file-type narrowPeak \
#     --rank p.value \
#     --output-file test \
#     --plot \
#     --log-output-file test.idr.log \
#     --verbose



#THIS WORKS
# idr --samples ERR3975815_ERR3975819_top100k_peaks.narrowPeak ERR3975833_ERR3975864_top100k_peaks.narrowPeak \
#     --input-file-type narrowPeak \
#     --rank p.value \
#     --output-file test3 \
#     --only-merge-peaks \
#     --log-output-file test3.idr.log \
#     --plot

# idr --samples ERR3975815_ERR3975819_top100k_peaks.narrowPeak ERR3975833_ERR3975864_top100k_peaks.narrowPeak \
#     --input-file-type narrowPeak \
#     --output-file test5.narrowPeak \
#     --rank p.value \
#     --plot