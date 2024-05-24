#!/bin/bash

###############################
#                             #
#    Snakemake Template       #
#                             #
###############################

######################
# 1.) Job Sumbission  
######################

# IMPORTANT NOTE: Snakemake has already been configured for you to be used with OSCAR and have certain default resources 
# These defaults were specified by you during installation and have been placed in a profile in Snakemake's configuration directory 
# However, Snakemake has also been configured to allow you to flexibly override the software defaults for any job
# IF you wish to override defaults, there are 2 ways (pick ONLY one): 
#    1. Create your own profile and use that in snakemake command (more work)
#    2. Specify desired HPC resources per task directly in your Snakefile (much easier) 

# Resources for base job that submits other jobs - if job gets killed due to limited resources, increase resources as needed 
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --mem=8g

#############
# 2.) Setup  
#############

# Enter Snakemake virtual environment module so you can run Snakemake commands 
snakemake_start

########################
# 3.) Running Snakemake 
########################

# Below, we include a rough example of how to run a Snakemake pipeline. 
# Please note that you will need a Snakefile (either created by you or someone else) that defines your workflow to run this script. 
# Also please note that the example below serves as starting guide and may not be exact. 
# Therefore, you might need to add/change quite a few things in order to get your specific pipeline/project to run. 

# HOW TO USE THIS TEMPLATE SCRIPT: 
# 1. First save this script under a different name and put it where needed (so you do not overwrite the template script)
# 2. Edit the example code below as needed and make sure code you want is uncommented (only ONE snakemake -s command should be ran!)
# 3. Lastly, save this script again (so you capture all your edits) and run this script via the sbatch command. That's it! 

# OVERRIDING DEFAULTS: If you wish to override the pre-configured software defaults and customize HPC resources yourself, 
# create your own profile and then replace the oscar profile with yours by using: -profile /path/to/your/profile_folder
# Another, and simpler, way to override defaults is to specify the HPC resources you want for each task directly in your Snakefile
# However, as a reminder, configuration has been handled by the installer for you, so you do NOT need to do either of these. 

### CODE EXAMPLE ###

# NOTE: You do NOT have to change the -profile oscar part of the command below unless you want to override defaults! 

# General example: 
snakemake -s /path/to/snakefile -profile oscar
# More specific example (made up): 
snakemake -s ${HOME}/snakemake_tutorial/tutorial.nf -profile oscar 


