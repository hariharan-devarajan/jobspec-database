#!/bin/bash 

coresxcpu=28     #to adjust according to the machine
rm -rf "${0%.*}" #remove pre-existing dir
mkdir  "${0%.*}" #make a dir with the same name as the executable
cd "${0%.*}"

for ncores in 1 2 4 8 16 28 
do

nnodes=$((ncores / coresxcpu))
remainder=$((ncores % coresxcpu))
if [ $remainder != 0 ]; then nnodes=$((nnodes+1)); fi #nnodes= number of nodes needed
 
mkdir ncores_$(printf %03d $ncores)                   #make dir like "ncores_008"
cd ncores_$(printf %03d $ncores)

cat > gauss.com << EOF
%NprocShared=$ncores
%Chk=gauss.chk
%Mem=50GB
#n CCSD-T/Def2TZVPP 

Dichloromethane scf

0 1
C         -1.07673        1.09508        0.08555
Cl         0.70112        1.11024        0.06910
Cl        -1.69042        2.22628       -1.14123
H         -1.43992        0.07150       -0.14064
H         -1.43992        1.40336        1.08746


EOF

cat > run.slurm << EOF
#!/bin/bash -l
#SBATCH --nodes=$nnodes
#SBATCH --ntasks=$ncores
#SBATCH --mem=60GB
#SBATCH --time=00:30:00
#SBATCH --constraint=E5v4

source /ssoft/spack/bin/slmodules.sh -r stable             
 
module load gaussian/g09-D.01
. \$g09root/g09/bsd/g09.profile

date_start=\$(date +%s)
\$g09root/g09/g09 < gauss.com > gauss.log           #running command            
date_end=\$(date +%s)
time_run=\$((date_end-date_start))
echo "$(printf %03d $ncores)_cpus \$time_run seconds"

rm -rf *chk        #remove useless big files (please!) 
EOF

sbatch run.slurm

cd ..

done
