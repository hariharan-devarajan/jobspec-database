#!/usr/bin/env bash
#SBATCH --job-name avg_ins_picard
#SBATCH --partition=fast
#SBATCH --time=00:10:00          # total run time limit in HH:MM:SS

#Cargamos los módulos y el programa:
module use /beegfs/easybuild/CentOS/7.6.1810/Skylake/modules/all
module use /beegfs/easybuild/common/modules/all
module load picard/2.26.10-Java-15
module load R/4.2.0-foss-2021b

cd /home/ireneg/DATOS_DNA-SEQ/procesamiento/ #Nos desplazamos a la carpeta con las carpetas de las muestras
for folder in $(ls -d C*/); do #Recorremos las carpetas de las muestras
        cd $folder #Entramos en la carpeta
        file="${folder:0:18}" #Obtenemos el nombre de la muestra
        java -jar $EBROOTPICARD/picard.jar CollectInsertSizeMetrics -I BWA/bwa_map_${file}_merge_PE_SE.sam -O BWA/output_metrics_bwa_map_${file}_merge_PE_SE.sam.txt -H BWA/insert_size_histogram_bwa_map_${file}_merge_PE_SE.sam.pdf -M 0.5 #Obtenemos el tamaño de inserción usando Picard
        cd .. #Salimos de la carpeta de la muestra
done

#En el comando de ejecución de Picard:
#-I especifica el archivo de entrada del que se va a obtener el tamaño de inserción
#-O especifica el archivo de salida en el que se van a almacenar las métricas de tamaño de inserción.
#-H indica el archivo de salida para un histograma del tamaño de inserción
#-M especifica que, al generar el histograma, se descartan las categorías de datos (fuera de FR, TANDEM, RF) que tengan menos de este porcentaje de lecturas totales. 0.5 indica que se incluirán en el cálculo las inserciones que representan al menos el 50% de los pares de lectura. Este valor es el que se establece en la documentación.
