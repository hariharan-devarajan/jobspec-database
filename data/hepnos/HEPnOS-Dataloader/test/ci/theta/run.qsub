#!/bin/bash
#COBALT -A CSC250STDM12
#COBALT -n 2
#COBALT -t 0:10:00
#COBALT --mode script
#COBALT -q debug-flat-quad

set -eu

HERE=`basename "$0"`

CONFIGFILE=$HERE/config.yaml
CONNECTIONFILE=$HERE/connection.yaml
INPUTFILE=$HERE/datafiles.txt
PDOMAIN=hepnosdloader
export MPICH_GNI_NDREG_ENTRIES=1024
export HOME=`pwd`

rm -rf ${CONNECTIONFILE}

echo "Setting up spack and modules"
source `pwd`/spack/share/spack/setup-env.sh
export MODULEPATH=$MODULEPATH:`pwd`/spack/share/spack/modules/cray-cnl6-mic_knl
spack load -r hepnos

echo "Setting up protection domain"
apstat -P | grep ${PDOMAIN} || apmgr pdomain -c -u ${PDOMAIN}

echo "Starting up HEPnOS daemon"
aprun -n 1 -N 1 -cc none -p ${PDOMAIN} hepnos-daemon $CONFIGFILE $CONNECTIONFILE &

echo "Waiting for HEPnOS daemon to start up"
while [ ! -f ${CONNECTIONFILE} ]; do sleep 10; done

echo "Starting dataloader"
echo `pwd`
time aprun -n 1 -N 1 -cc none -p ${PDOMAIN} ./build/dataloader ${CONNECTIONFILE} ${INPUTFILE} debug

echo "Shutting down HEPnOS"
aprun -n 1 -N 1 -cc none -p ${PDOMAIN} hepnos-shutdown $CONNECTIONFILE

sleep 5
echo "Removing protection domain"
apmgr pdomain -r -u ${PDOMAIN}
