#!/bin/bash
#
#-----------------------------------------------------------------------------
# This Lonestar6 job script is designed to create a vnc session on 
# visualization nodes through the SLURM batch system. Once the job
# is scheduled, check the output of your job (which by default is
# stored in your home directory in a file named vncserver.out)
# and it will tell you the port number that has been setup for you so
# that you can attach via a separate VNC client to any Lonestar6 login 
# node (e.g., login1.ls6.tacc.utexas.edu).
#
# Note that for security, we REQUIRE setting up a tunneled VNC
# session in order to connect via a client (more information on doing
# this is available at the User Guide link below).  Once you connect,
# you should see a single xterm running which you can use to launch
# any X application (eg. ParaView or VisIt) 
#
# Note: you can fine tune the SLURM submission variables below as
# needed.  Typical items to change are the runtime limit, location of
# the job output, and the allocation project to submit against (it is
# commented out for now, but is required if you have multiple
# allocations).  
#
# To submit the job, issue: "sbatch /share/doc/slurm/job.vnc" 
#
# For more information, please consult the User Guide at: 
#
# https://portal.tacc.utexas.edu/user-guides/lonestar6
#-----------------------------------------------------------------------------
#
#SBATCH -J vncserver                  # Job name
#SBATCH -o vncserver.out              # Name of stdout output file (%j expands to jobId)
#SBATCH -p development                # Queue name
#SBATCH -N 2                          # Total number of nodes requested (68 cores/node)
#SBATCH -n 2                          # Total number of mpi tasks requested
#SBATCH -t 02:00:00                   # Run time (hh:mm:ss) - 4 hours

#--------------------------------------------------------------------------
# ---- You normally should not need to edit anything below this point -----
#--------------------------------------------------------------------------
#
# last update: pnav 20220126

TAP_FUNCTIONS="/share/doc/slurm/tap_functions"
if [ -f ${TAP_FUNCTIONS} ]; then
    . ${TAP_FUNCTIONS}
else
    echo "TACC:"
    echo "TACC: ERROR - could not find TAP functions file: ${TAP_FUNCTIONS}"
    echo "TACC: ERROR - Please submit a consulting ticket at the TACC user portal"
    echo "TACC: ERROR - https://portal.tacc.utexas.edu/tacc-consulting/-/consult/tickets/create"
    echo "TACC:"
    echo "TACC: job $SLURM_JOB_ID execution finished at: `date`"
    exit 1
fi

echo "TACC: job ${SLURM_JOB_ID} execution at: $(date)"

# our node name
NODE_HOSTNAME=$(hostname -s)
echo "TACC: running on node ${NODE_HOSTNAME}"

# VNC server executable
VNCSERVER_BIN=$(which vncserver 2> /dev/null)
echo "TACC: using default VNC server ${VNCSERVER_BIN}"

if [ -z ${VNCSERVER_BIN} ]; then
    echo "TACC:" 
    echo "TACC: ERROR - could not find the vncserver"
    echo "TACC: ERROR - please submit a ticket at the TACC User Portal."
    echo "TACC: ERROR - https://portal.tacc.utexas.edu/tacc-consulting/-/consult/tickets/create"
    echo "TACC:"
    echo "TACC: job ${SLURM_JOB_ID} execution finished at: $(date)"
    exit 1
fi

# Check whether a vncpasswd file exists.  If not, complain and exit.
if [ ! -f ${HOME}/.vnc/passwd ] ; then
    echo "TACC:" 
    echo "TACC: ERROR - You must run 'vncpasswd' once before launching a vnc session"
    echo "TACC:"
    echo "TACC: job ${SLURM_JOB_ID} execution finished at: $(date)"
    exit 1
fi

CONDA=$(which conda 2> /dev/null)
if [ ! -z "${CONDA}" ]; then
    CONDA_ENV=$(conda info | grep active | cut -d ":" -f 2)
    if [[ ! "${CONDA_ENV}" =~ "None" ]]; then 
        echo "TACC:"
        echo "TACC: ERROR - active conda installation detected, which will break VNC"
        echo "TACC: ERROR - deactivate conda with 'conda deactivate'"
        echo "TACC: ERROR - then resubmit this job script"
        echo "TACC: ERROR - Questions? Please submit a consulting ticket"
        echo "TACC: ERROR - https://portal.tacc.utexas.edu/tacc-consulting/-/consult/tickets/create"
        echo "TACC:"
        echo "TACC: job ${SLURM_JOB_ID} execution finished at: $(date)"
        exit 1
    fi
fi

# launch VNC session
VNC_ARGS="-localhost $@"
VNC_DISPLAY=`${VNCSERVER_BIN} ${VNC_ARGS} 2>&1 | grep desktop | awk -F: '{print $3}'`
echo "TACC: got VNC display :${VNC_DISPLAY}"

if [ -z "${VNC_DISPLAY}" ]; then
    echo "TACC:"
    echo "TACC: ERROR - Error launching vncserver: ${VNCSERVER}"
    echo "TACC: Please submit a ticket to the TACC User Portal"
    echo "TACC: https://portal.tacc.utexas.edu/"
    echo "TACC:"
    echo "TACC: job ${SLURM_JOB_ID} execution finished at: $(date)"
    exit 1
fi

LOCAL_VNC_PORT=$(( 5900 + ${VNC_DISPLAY}))
echo "TACC: local (compute node) VNC port is $LOCAL_VNC_PORT"

LOGIN_PORT=$(tap_get_port)
echo "TACC: got login node VNC port $LOGIN_PORT"

# create reverse tunnel port to login nodes.  Make one tunnel for each login so the user can just
# connect to ls6.tacc
TUNNEL_CMD="ssh -q -g -f -N -o StrictHostKeyChecking=no -L localhost:${LOGIN_PORT}:localhost:${LOCAL_VNC_PORT} ${NODE_HOSTNAME}"
for i in $(seq 3); do
    ssh -q -o StrictHostKeyChecking=no login$i "${TUNNEL_CMD}" &
done
echo "TACC: created reverse ports on Lonestar6 logins"

echo "TACC: Your VNC server is now running!"
echo "TACC: To connect via VNC client, create a local ssh tunnel to Lonestar6 using:"
echo "TACC:     ssh -f -N -L ${LOGIN_PORT}:localhost:${LOGIN_PORT} ${USER}@ls6.tacc.utexas.edu"
echo "TACC: Then connect your vncviewer to localhost:${LOGIN_PORT}"

# set display for X applications
export DISPLAY=":${VNC_DISPLAY}"

# we need vglclient to run to have graphics across multi-node jobs
vglclient >& /dev/null &
VGL_PID=$!

mkdir -p ${HOME}/.tap # this should exist at this point, but just in case...
TAP_LOCKFILE=${HOME}/.tap/${SLURM_JOB_ID}.lock
DISPLAY=:1 xterm -fg white -bg red3 +sb -geometry 55x2+0+0 -T 'END SESSION HERE' -e "echo 'TACC: Press <enter> in this window to end your session' && read && rm ${TAP_LOCKFILE}" &
sleep 1
DISPLAY=:1 xterm -ls -geometry 80x24+100+50 &

# spin on lock until file is removed
echo $(date) > ${TAP_LOCKFILE}
while [ -f ${TAP_LOCKFILE} ]; do
    sleep 1
done


# job is done!
echo "TACC: Killing VGL client"
kill $VGL_PID

echo "TACC: Killing VNC server" 
vncserver -kill $DISPLAY

echo "TACC: release port returned $(tap_release_port ${LOGIN_PORT} 2> /dev/null)"

# wait a brief moment so vncserver can clean up after itself
sleep 1

echo "TACC: job ${SLURM_JOB_ID} execution finished at: $(date)"

