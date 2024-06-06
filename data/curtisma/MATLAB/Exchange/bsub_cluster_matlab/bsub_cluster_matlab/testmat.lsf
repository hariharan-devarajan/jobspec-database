#!/bin/bash

# enable your environment, which will use .bashrc configuration in your home directory
#BSUB -L /bin/bash

# the name of your job showing on the queue system
#BSUB -J myjob

# the following BSUB line specify the queue that you will use,
# please use bqueues command to check the available queues for each toolbox of matlab
# Bioinformatics toolbox, please use matlabbio queue
# Signal Processing toolbox, please use matlabsig queue
# Image Processing toolbox, please use matlabimg queue
# Wavelet toolbox, please use matlabwav queue
# Matlab DCE, please use matlabdce queue (or matlabdce-short for very small jobs like this one)
# Matlab, Optimization and Statistic, please use matlab queue


#BSUB -q matlabdce-short


# the system output and error message output, %J will show as your jobID
#BSUB -o %J.out
#BSUB -e %J.err

#the CPU number that you will collect (Attention: each node has 2 CPU)
#BSUB -n 1


#when job finish that you will get email notification
#BSUB -u YourEmail@email.com
#BSUB -N


# Finally, Start the program
matlab < myplot.m

