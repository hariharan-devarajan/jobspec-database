#!/bin/bash

###############################
#                             #
#    Nextflow Template        #
#                             #
###############################

#######################
# 1.) Job Sumbission  
#######################

# IMPORTANT NOTE: Nextflow has already been configured for you to be used with OSCAR and have certain default resources 
# These defaults were specified by you during installation and have been placed in a config.yaml file in Nextflow's configuration path
# However, Nextflow has also been configured to allow you to flexibly add to or override the software defaults for any job you run
# IF you wish to override defaults, there are 2 ways (pick ONLY one): 
#    1. Specify desired HPC resources in your .nf pipeline file 
#    2. Specify your desired HPC resources in your own config.yaml file and call that during nextflow run (see Running Nextflow section below)

# Resources for base job that submits other jobs - edit email here; if job gets killed due to limited resources, increase resources as needed 
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --mem=8g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=enter_email_here@brown.edu   # <----- EDIT this to be your Brown email, but do not remove the # signs here  


#############
# 2.) Setup  
#############

# Set environment vairables (you do not need to change these!) 
export SINGULARITY_CACHEDIR=$HOME/scratch
export NXF_SINGULARITY_CACHEDIR=$HOME/scratch

# Enter Nextflow virtual environment module so you can run Nextflow commands 
nextflow_start

########################
# 3.) Running Nextflow 
########################

# Now you can run your nextflow pipeline (please read all comments below!)

# There are three ways to run a nextflow pipeline on OSCAR: 
# 1.) run a pipeline from nf-core; 
# 2.) run a pipeline from a github repo; 
# 3.) run your own custom pipeline

# Below, we include rough examples for how to run a Nextflow pipeline for each of the three ways listed above.
# Please note that the examples below serve as starting guides and may not be exact. 
# Therefore, you might need to add/change quite a few things in order to get your specific pipeline/project to run. 

# HOW TO USE THIS TEMPLATE SCRIPT: 
# 1. First save this script under a different name and put it where needed (so you do not overwrite the template script)
# 2. Choose the example below that fits your situation and edit that example according to your specific needs
# 3. Leave your edited, new nextflow run command in the script and make sure it is uncommented 
# 4. Delete (or comment out) all the other code examples 
# 5. Lastly, save this script again (so you capture all your edits) and run this script via the sbatch command. That's it! 

# OVERRIDING DEFAULTS: If you wish to override the pre-configured software defaults and customize HPC resources yourself, 
# just include the following code to the end of any of the commands below: -c /path/to/your/config.yaml
# However, as a reminder, this has been handled by the installer for you, so you do NOT need to do this. 

# NOTE: You might need to use the -profile singularity argument with pipelines other than pipelines from nf-core (Example 1).
# The use of -profile in the nextflow run command depends on how the pipeline is built and configured, so please make sure 
# to look into pipeline documentation. If this argument is required, just add it to the end of your nextflow run command. 

### EXAMPLE 1: nf-core ###

# General example: 
nextflow run nf-core/<name_of_nf-core_pipeline_you_want> -profile singularity 
# More specific example (from https://nf-co.re/ampliseq)
nextflow run nf-core/ampliseq -profile singularity 

### EXAMPLE 2: GitHub ###

# General example: 
nextflow run nextflow-io/<name_of_pipeline_you_want> 
# More specific example (from https://github.com/nextflow-io/rnatoy):  
nextflow run nextflow-io/rnatoy  

### EXAMPLE 3: Your own personally developed pipeline (you need your own .nf file) ###

# General example: 
nextflow run <path_to_your_nextflow.nf_file_and_project_directory> 
# More specific example (made up): 
nextflow run ${HOME}/nextflow_tutorial/tutorial.nf 
