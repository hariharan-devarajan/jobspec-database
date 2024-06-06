#!/usr/bin/env bash
#SBATCH --mem=200GB
#SBATCH --job-name MaSuRCA
#SBATCH --partition=long
#SBATCH --time=1-59:00:20          # total run time limit in HH:MM:SS

#Cargamos los módulos 
module use /beegfs/easybuild/CentOS/7.6.1810/Skylake/modules/all
module use /beegfs/easybuild/common/modules/all
#Activamos nuestro usuario de mamba para poder acceder a MaSuRCA
mamba init
mamba activate irenegmasurca

cd /home/ireneg/DATOS_DNA-SEQ/procesamiento/ #Nos desplazamos a la carpeta con las carpetas de las muestras
for folder in $(ls -d C*/); do #Recorremos las carpetas de las muestras, en las que se encuentran las lecturas de la secuenciación
        cd $folder #Entramos en la carpeta
        file="${folder:0:18}" #Obtenemos el nombre de la muestra
        file_1="${file}_1.fastq.gz" #Obtenemos el nombre del archivo con las lecturas forward
        file_2="${file}_2.fastq.gz" #Obtenemos el nombre del archivo con las lecturas reverse 
        masurca -t 32 -i ${file_1},${file_2} #Ejecutamos el ensamblado con las lecturas originales
        cd .. #Salimos de la carpeta de la muestra
done

#En el comando de ejecución de MaSuRCA:
#-t especifica el número de threads utilizados en la ejecución
#-i indican los archivos con las lecturas a mapear. MaSuRCA trabaja con lecturas sin filtrar, por lo que son estas las que debemos aportar. Para especificar que se trabaja con lecturas emparejadas, se separan los archivos forward y reverse con una coma.
