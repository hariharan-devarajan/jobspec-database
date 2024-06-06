#!/usr/bin/env bash
#SBATCH --mem=200GB
#SBATCH --job-name SPAdes
#SBATCH --partition=medium
#SBATCH --time=24:00:00  

#Cargamos los módulos y el programa:
module use /beegfs/easybuild/CentOS/7.6.1810/Skylake/modules/all
module use /beegfs/easybuild/common/modules/all
module load SPAdes/3.15.2-GCC-8.2.0-2.31.1

cd /home/ireneg/DATOS_DNA-SEQ/fastp+segemehl+bwa/ #Nos desplazamos a la carpeta con las carpetas de las muestras
for folder in $(ls -d C*/); do #Recorremos las carpetas de las muestras
        cd $folder #Entramos en la carpeta
        file="${folder:0:18}" #Obtenemos el nombre de la muestra
        spades.py --pe1-1 out_fastp_${file}_1_paired.fq --pe1-2 out_fastp_${file}_2_paired.fq --pe1-s out_fastp_${file}_1_unpaired.fq --pe1-s out_fastp_${file}_2_unpaired.fq -o spades_output #Ejecutamos el ensamblado con SPAdes
        cd .. #Salimos de la carpeta de la muestra
done

#En el comando de ejecución de SPAdes se permiten incluir usando --pe1-1, --pe1-2, --pe1-s, --pe1-s, las lecturas emparejadas forward, las lecturas emparejadas reverse, las lecturas desemparejadas forward y las lecturas desemparejadas reverse, respectivamente
#-o se indica para determinar la carpeta en la que se almacenará el output del programa
