#BSUB -n 12
#BSUB -J render
#BSUB -e logs/err.txt
#BSUB -o logs/out.txt
#BSUB -R "rusage[mem=128] span[hosts=1]"

mkdir -p logs

#singularity image from dockerhub rnabioco/raer-ms
image="../../docker/raer-ms.sif"

cd dbases
bash get_databases.sh
cd -

rmd="rediportal-data-processing.R"
singularity exec $image R --vanilla -e  "source('${rmd}')"

rmd="10x-data-processing.Rmd"
singularity exec $image R --vanilla -e  "rmarkdown::render('${rmd}')"

rmd="GSE99249-data-processing.Rmd"
singularity exec $image R --vanilla -e  "rmarkdown::render('${rmd}')"

rmd="NA12877-data-processing.Rmd"
singularity exec $image R --vanilla -e  "rmarkdown::render('${rmd}')"
