#!/bin/bash
#SBATCH -n 6      
#SBATCH -N 1                
#SBATCH -t 2880              
#SBATCH -p general
#SBATCH --mem 100000    
#SBATCH -o hostname2.out    
#SBATCH -e hostname2.err    
#SBATCH --mail-type=ALL     
#SBATCH --mail-user=sywang1984@gmail.com  
module load bio/OligoArray2_1
java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence15.fasta -d chr22.fa -o oligos15.txt -r failed15.txt -R log15.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence16.fasta -d chr22.fa -o oligos16.txt -r failed16.txt -R log16.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence17.fasta -d chr22.fa -o oligos17.txt -r failed17.txt -R log17.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence18.fasta -d chr22.fa -o oligos18.txt -r failed18.txt -R log18.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence19.fasta -d chr22.fa -o oligos19.txt -r failed19.txt -R log19.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence20.fasta -d chr22.fa -o oligos20.txt -r failed20.txt -R log20.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence21.fasta -d chr22.fa -o oligos21.txt -r failed21.txt -R log21.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence22.fasta -d chr22.fa -o oligos22.txt -r failed22.txt -R log22.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence23.fasta -d chr22.fa -o oligos23.txt -r failed23.txt -R log23.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence24.fasta -d chr22.fa -o oligos24.txt -r failed24.txt -R log24.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence25.fasta -d chr22.fa -o oligos25.txt -r failed25.txt -R log25.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence26.fasta -d chr22.fa -o oligos26.txt -r failed26.txt -R log26.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

java -Xmx6g -jar /n/sw/OligoArray2_1/OligoArray2.jar -i 1kb_fragments_sequence27.fasta -d chr22.fa -o oligos27.txt -r failed27.txt -R log27.txt -n 33 -l 30 -L 30 -D 1000 -t 60 -T 100 -s 70 -x 70 -p 30 -P 90 -m "GGGGGGG;CCCCCCC;TTTTTTT;AAAAAAA" -g 31 -N 6

