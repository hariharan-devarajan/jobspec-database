nameId=HIG-RunIIFall18wmLHEGS-02758
#nameId=HIG-RunIISummer15wmLHEGS-02758
#nameId=HIG-RunIISummer15wmLHEGS-02710
#nameId=HIG-RunIIFall17wmLHEGS-04016
#nameId=HIG-RunIIFall17wmLHEGS-04015
#nameId=HIG-RunIIFall18wmLHEGS-00581
#nameId=HIG-RunIIFall17wmLHEGS-02473
#nameId=HIG-RunIIFall17wmLHEGS-01941
#nameId=HIG-RunIIFall17wmLHEGS-01944
#nameId=HIG-RunIIFall17wmLHEGS-01709
wget https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/${nameId}

#wget https://cms-pdmv.cern.ch/mcm/public/restapi/requests/get_test/HIG-RunIIFall17wmLHEGS-01668
chmod u+x ${nameId}
##################################################################################
# If you  want to run this interactivley, you should run this 
# outside of CMSSW.
#
# Inside the script, you can set the number of generated events
# trace this number up as in "echo 31 event", then set the number as you like
##################################################################################
#
#
# Important to use batch job
# please put these lines to specify the full path of working directory
# #!/bin/bash
# cd /afs/cern.ch/user/s/salee/tmp/ggHWWTo2L2Nu
# source /cvmfs/cms.cern.ch/cmsset_default.sh
# export SCRAM_ARCH=slc6_amd64_gcc630

#bsub -q 8nh ${nameId}

