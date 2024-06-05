#!/bin/bash
#COBALT -A HEP_on_HPC
#COBALT -n 128
#COBALT -t 2:00:00
#COBALT --mode script

set -eu

CONFIGFILE=config.yaml
CONNECTIONFILE=connection.yaml
INPUTFILE=datafiles.txt
HERE=`basename "$0"`
PDOMAIN=hepnostest
export MPICH_GNI_NDREG_ENTRIES=1024

rm -f ${CONNECTIONFILE}

echo "Setting up spack and modules"
source $HOME/spack/share/spack/setup-env.sh
export MODULEPATH=$MODULEPATH:/spack/share/spack/modules/cray-cnl6-mic_knl
spack env activate hepnos
spack load -r hepnos

echo "Setting up protection domain"
apstat -P | grep ${PDOMAIN} || apmgr pdomain -c -u ${PDOMAIN}

echo "Starting up HEPnOS daemon"
aprun -n 16 -N 1 -cc none -d 64 -p ${PDOMAIN} hepnos-daemon $CONFIGFILE $CONNECTIONFILE &

echo "Waiting for HEPnOS daemon to start up"
while [ ! -f ${CONNECTIONFILE} ]; do sleep 10; done

echo "Starting dataloader"
echo `pwd`
time aprun -n 112 -N 1 -cc none -d 64 -p ${PDOMAIN} ../build/dataloader \
	     -c ${CONNECTIONFILE} \
	     -i ${INPUTFILE} \
	     -o Nova \
	     -l xxx \
	     -a \
	     -t 1 \
	     -b 1024 \
	     -v debug

echo "Shutting down HEPnOS"
aprun -n 1 -N 1 -cc none -p ${PDOMAIN} hepnos-shutdown $CONNECTIONFILE

sleep 5
echo "Destroying protection domain"
apmgr pdomain -r -u ${PDOMAIN}
