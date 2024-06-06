#PBS -N Han_script
#PBS -o matlab_output.log        # Standard output file
#PBS -e matlab_error.log         # Standard error file
#PBS -l nodes=1:ppn=40           # Request 1 node with 40 processors
cd $PBS_O_WORKDIR || exit 1
/usr/local/MATLAB/R2022a/bin/matlab -nodisplay -nodesktop -nosplash -batch parallel_script_geometry_material_properties