#PBS -l nodes=1:ppn=1:haswell,walltime=24:00:00,mem=8gb
#PBS -o <Processing Folder>
#PBS -e <Processing Folder>

module load singularity-3.2.1

singularity exec -B <directory of run_qunex.sh>,<directory of parameter file>,<directory of gradient_coefficient_files>:/export/HCP/gradient_coefficient_files <path to qunex oontainer/qunex.sif> <directory of run_qunex.sh>/run_qunex.sh \
  --parameterfolder=<directory of parameter file> \
  --studyfolder=<Processing Folder>/<subject_name> \
  --subjects=<subject_name> \
  --scan=<scan_name> \
  --overwrite=yes \
  --hcppipelineprocess=FunctionalPreprocessing
