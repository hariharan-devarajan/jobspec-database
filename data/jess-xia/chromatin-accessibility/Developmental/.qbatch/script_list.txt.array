#!/bin/sh
#SBATCH --nodes=1
#SBATCH 
#SBATCH --output=/external/rprshnas01/kcni/jxia/chromatin-accessibility/Developmental/logs/slurm-script_list.txt-%A_%a.out
#SBATCH --chdir=/external/rprshnas01/kcni/jxia/chromatin-accessibility/Developmental
#SBATCH --job-name=script_list.txt
#SBATCH 
#SBATCH 
#SBATCH --array=1-337
#SBATCH --time=1:00:00
#SBATCH 
#SBATCH 
#SBATCH --export=NONE
# -- start copied env
export __Init_Default_Modules="1"
export TMOUT="7200"
export QT_GRAPHICSSYSTEM_CHECKED="1"
export SACCT_FORMAT="JobID%20,JobName,User,Partition,NodeList,Elapsed,State,ExitCode,MaxRSS,AllocTRES%32"
export LMOD_ANCIENT_TIME="86400"
export SHELL="/bin/bash"
export EBVERSIONMINICONDA3="4.9.2"
export HISTSIZE="1000"
export MANPATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/share/man:/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/man:/opt/scc/conda/software/Miniconda3/4.9.2/share/man:/usr/local/apps/lmod/lmod/share/man::"
export XDG_RUNTIME_DIR="/run/user/10355"
export SQUEUE_FORMAT="%20i %20u %20a %20P %10Q %5D %5C %11l %11L %18R %40j"
export FPATH="/usr/local/apps/lmod/lmod/init/ksh_funcs"
export __LMOD_REF_COUNT_MODULEPATH="/usr/local/scc/modulefiles/all:1;/usr/local/scc/modulefiles/quarantine:1;/usr/local/scc/modulefiles/other:1;/usr/local/apps/lmod/8.4.20/modulefiles/Core:1;/etc/modulefiles:1"
export LMOD_SETTARG_FULL_SUPPORT="no"
export CONDA_ENV="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11"
export XDG_SESSION_ID="43048"
export HOSTNAME="scclogin01.camhres.ca"
export LMOD_ROOT="/usr/local/apps/lmod"
export _ModuleTable002_="MDAwMDAzLjAwMDAwMDAwOC4wMDAwMDAwMDUuKmFuYWNvbmRhLjAwMDAwMDAwMy4qemZpbmFsLS4wMDAwMDIwMjAuMDAwMDAwMDExLip6ZmluYWwiLH0sU0NDPXtbImZuIl09Ii91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVzL290aGVyL1NDQy5sdWEiLFsiZnVsbE5hbWUiXT0iU0NDIixbImxvYWRPcmRlciJdPTEscHJvcFQ9e30sWyJzdGFja0RlcHRoIl09MCxbInN0YXR1cyJdPSJhY3RpdmUiLFsidXNlck5hbWUiXT0iU0NDIixbIndWIl09Ik0uKnpmaW5hbCIsfSxbImxhbmcvTWluaWNvbmRhMyJdPXtbImZuIl09Ii91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVzL2FsbC9sYW5nL01pbmljb25kYTMvNC45LjIubHVhIixbImZ1bGxOYW1lIl09ImxhbmcvTWluaWNvbmRhMy80Ljku"
export LMOD_PACKAGE_PATH="/usr/local/apps/lmod/etc"
export MAIL="/var/spool/mail/jxia"
export LS_COLORS="rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=01;05;37;41:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.axv=01;35:*.anx=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=01;36:*.au=01;36:*.flac=01;36:*.mid=01;36:*.midi=01;36:*.mka=01;36:*.mp3=01;36:*.mpc=01;36:*.ogg=01;36:*.ra=01;36:*.wav=01;36:*.axa=01;36:*.oga=01;36:*.spx=01;36:*.xspf=01;36:"
export CONDA_PYTHON_EXE="/opt/scc/conda/software/Miniconda3/4.9.2/bin/python"
export __LMOD_REF_COUNT_MANPATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/share/man:1;/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/man:1;/opt/scc/conda/software/Miniconda3/4.9.2/share/man:1;/usr/local/apps/lmod/lmod/share/man:1"
export _CE_M=""
export LESSOPEN="||/usr/bin/lesspipe.sh %s"
export USER="jxia"
export EBROOTMINICONDA3="/opt/scc/conda/software/Miniconda3/4.9.2"
export __LMOD_REF_COUNT_LD_LIBRARY_PATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib/R/lib:1;/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib:1;/usr/local/apps/software/Tcl/8.5.13/lib:1;/usr/local/apps/software/zlib/1.2.11/lib:1;/usr/local/apps/software/Lua/5.1.4-8/lib:1"
export SHLVL="1"
export DISPLAY="localhost:12.0"
export LMOD_ADMIN_FILE="/usr/local/apps/lmod/etc/admin.list"
export QBATCH_SYSTEM="slurm"
export MODULESHOME="/usr/local/apps/lmod/lmod"
export __LMOD_REF_COUNT_PKG_CONFIG_PATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib/pkgconfig:1;/opt/scc/conda/software/Miniconda3/4.9.2/lib/pkgconfig:1"
export _ModuleTable001_="X01vZHVsZVRhYmxlXz17WyJNVHZlcnNpb24iXT0zLFsiY19yZWJ1aWxkVGltZSJdPWZhbHNlLFsiY19zaG9ydFRpbWUiXT1mYWxzZSxkZXB0aFQ9e30sZmFtaWx5PXt9LG1UPXtSPXtbImZuIl09Ii91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVzL3F1YXJhbnRpbmUvUi80LjAuMy1QeXRob24tMy44LjUtQW5hY29uZGEzLTIwMjAuMTEubHVhIixbImZ1bGxOYW1lIl09IlIvNC4wLjMtUHl0aG9uLTMuOC41LUFuYWNvbmRhMy0yMDIwLjExIixbImxvYWRPcmRlciJdPTMscHJvcFQ9e30sWyJzdGFja0RlcHRoIl09MCxbInN0YXR1cyJdPSJhY3RpdmUiLFsidXNlck5hbWUiXT0iUiIsWyJ3ViJdPSIwMDAwMDAwMDQuMDAwMDAwMDAwLjAwMDAwMDAwMy4qeXRob24uKnpmaW5hbC0uMDAw"
export LMOD_VERSION="8.4.20"
export PKG_CONFIG_PATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib/pkgconfig:/opt/scc/conda/software/Miniconda3/4.9.2/lib/pkgconfig"
export BASH_ENV="/usr/local/apps/lmod/lmod/init/bash"
export __LMOD_REF_COUNT_PATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11:1;/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/sbin:1;/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/bin:1;/opt/scc/conda/software/Miniconda3/4.9.2:1;/opt/scc/conda/software/Miniconda3/4.9.2/bin:1;/usr/local/apps/bin:1;/usr/local/apps/sbin:1;/lmod/lmod/libexec:1;/opt/slurmtools/bin:1;/usr/local/bin:1;/usr/bin:1;/usr/local/sbin:1;/usr/sbin:1;/sbin:1;/bin:1"
export _="/usr/bin/qbatch"
export MODULEPATH="/usr/local/scc/modulefiles/all:/usr/local/scc/modulefiles/quarantine:/usr/local/scc/modulefiles/other:/usr/local/apps/lmod/8.4.20/modulefiles/Core:/etc/modulefiles"
export LMOD_SHORT_TIME="86400"
export SSH_CONNECTION="172.28.201.185 60377 172.26.216.50 22"
export EBVERSIONR="4.0.3"
export OMP_NUM_THREADS="1"
export _LMFILES_="/usr/local/scc/modulefiles/other/SCC.lua:/usr/local/scc/modulefiles/all/lang/Miniconda3/4.9.2.lua:/usr/local/scc/modulefiles/quarantine/R/4.0.3-Python-3.8.5-Anaconda3-2020.11.lua"
export __LMOD_REF_COUNT_LOADEDMODULES="SCC:1;lang/Miniconda3/4.9.2:1;R/4.0.3-Python-3.8.5-Anaconda3-2020.11:1"
export LMOD_DIR="/usr/local/apps/lmod/lmod/libexec"
export LMOD_PKG="/usr/local/apps/lmod/lmod"
export MODULEPATH_ROOT="/usr/local/apps/modulefiles"
export HOME="/nethome/kcni/jxia"
export LD_LIBRARY_PATH="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib/R/lib:/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/lib:/usr/local/apps/software/Tcl/8.5.13/lib:/usr/local/apps/software/zlib/1.2.11/lib:/usr/local/apps/software/Lua/5.1.4-8/lib"
export LANG="en_US.UTF-8"
export CONDA_SHLVL="0"
export CONDA_PREFIX="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11"
export __LMOD_REF_COUNT__LMFILES_="/usr/local/scc/modulefiles/other/SCC.lua:1;/usr/local/scc/modulefiles/all/lang/Miniconda3/4.9.2.lua:1;/usr/local/scc/modulefiles/quarantine/R/4.0.3-Python-3.8.5-Anaconda3-2020.11.lua:1"
export _ModuleTable_Sz_="4"
export CONDA_DEFAULT_ENV="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11"
export LMOD_CMD="/usr/local/apps/lmod/lmod/libexec/lmod"
export SCC_LMOD_HIDE="1"
export EBDEVELMINICONDA3="/opt/scc/conda/software/Miniconda3/4.9.2/easybuild/lang-Miniconda3-4.9.2-easybuild-devel"
export SSH_TTY="/dev/pts/12"
export OLDPWD="/nethome/kcni/jxia/chromatin-accessibility"
export LMOD_SYSTEM_DEFAULT_MODULES="SCC"
export LMOD_CACHED_LOADS="no"
export SSH_CLIENT="172.28.201.185 60377 22"
export LOGNAME="jxia"
export PATH="/opt/scc/conda/software/Miniconda3/4.9.2/condabin:/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11:/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/sbin:/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/bin:/opt/scc/conda/software/Miniconda3/4.9.2:/opt/scc/conda/software/Miniconda3/4.9.2/bin:/usr/local/apps/bin:/usr/local/apps/sbin:/lmod/lmod/libexec:/opt/slurmtools/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin"
export TERM="xterm"
export EBDEVELR="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11/easybuild/lang-R-4.0.3-Python-3.8.5-Anaconda3-2020.11-easybuild-devel"
export __LMOD_SET_FPATH="1"
export _ModuleTable003_="MiIsWyJsb2FkT3JkZXIiXT0yLHByb3BUPXt9LFsic3RhY2tEZXB0aCJdPTEsWyJzdGF0dXMiXT0iYWN0aXZlIixbInVzZXJOYW1lIl09ImxhbmcvTWluaWNvbmRhMy80LjkuMiIsWyJ3ViJdPSIwMDAwMDAwMDQuMDAwMDAwMDA5LjAwMDAwMDAwMi4qemZpbmFsIix9LH0sbXBhdGhBPXsiL3Vzci9sb2NhbC9zY2MvbW9kdWxlZmlsZXMvYWxsIiwiL3Vzci9sb2NhbC9zY2MvbW9kdWxlZmlsZXMvcXVhcmFudGluZSIsIi91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVzL290aGVyIiwiL3Vzci9sb2NhbC9hcHBzL2xtb2QvOC40LjIwL21vZHVsZWZpbGVzL0NvcmUiLCIvZXRjL21vZHVsZWZpbGVzIix9LFsic3lzdGVtQmFzZU1QQVRIIl09Ii91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVz"
export _CE_CONDA=""
export _ModuleTable004_="L2FsbDovdXNyL2xvY2FsL3NjYy9tb2R1bGVmaWxlcy9xdWFyYW50aW5lOi91c3IvbG9jYWwvc2NjL21vZHVsZWZpbGVzL290aGVyOi91c3IvbG9jYWwvYXBwcy9sbW9kLzguNC4yMC9tb2R1bGVmaWxlcy9Db3JlOi9ldGMvbW9kdWxlZmlsZXMiLH0="
export KRB5CCNAME="KEYRING:persistent:10355"
export LMOD_sys="Linux"
export LOADEDMODULES="SCC:lang/Miniconda3/4.9.2:R/4.0.3-Python-3.8.5-Anaconda3-2020.11"
export CONDA_EXE="/opt/scc/conda/software/Miniconda3/4.9.2/bin/conda"
export HISTCONTROL="ignoredups"
export EBROOTR="/opt/scc/conda/software/R/4.0.3-Python-3.8.5-Anaconda3-2020.11"
# -- end copied env

ARRAY_IND=$SLURM_ARRAY_TASK_ID

command -v parallel > /dev/null 2>&1 || { echo "GNU parallel not found in job environment. Exiting."; exit 1; }
CHUNK_SIZE=1
CORES=1
export THREADS_PER_COMMAND=1
sed -n "$(( (${ARRAY_IND} - 1) * ${CHUNK_SIZE} + 1 )),+$(( ${CHUNK_SIZE} - 1 ))p" << EOF | parallel -j${CORES} --tag --line-buffer --compress
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_1.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_2.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_3.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_4.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_5.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_6.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_7.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_8.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_9.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_10.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_11.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_12.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_13.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_14.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_15.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_16.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_17.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_18.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_19.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_20.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_21.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_22.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_23.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_24.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_25.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_26.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_27.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_28.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_29.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_30.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_31.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_32.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_33.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_34.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_35.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_36.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_37.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_38.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_39.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_40.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_41.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_42.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_43.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_44.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_45.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_46.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_47.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_48.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_49.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_50.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_51.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_52.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_53.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_54.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_55.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_56.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_57.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_58.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_59.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_60.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_61.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_62.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_63.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_64.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_65.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_66.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_67.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_68.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_69.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_70.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_71.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_72.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_73.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_74.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_75.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_76.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_77.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_78.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_79.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_80.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_81.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_82.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_83.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_84.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_85.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_86.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_87.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_88.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_89.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_90.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_91.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_92.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_93.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_94.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_95.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_96.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_97.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_98.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_99.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_100.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_101.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_102.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_103.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_104.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_105.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_106.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_107.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_108.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_109.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_110.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_111.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_112.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_113.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_114.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_115.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_116.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_117.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_118.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_119.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_120.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_121.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_122.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_123.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_124.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_125.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_126.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_127.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_128.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_129.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_130.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_131.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_132.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_133.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_134.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_135.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_136.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_137.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_138.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_139.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_140.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_141.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_142.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_143.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_144.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_145.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_146.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_147.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_148.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_149.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_150.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_151.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_152.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_153.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_154.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_155.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_156.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_157.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_158.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_159.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_160.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_161.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_162.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_163.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_164.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_165.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_166.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_167.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_168.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_169.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_170.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_171.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_172.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_173.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_174.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_175.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_176.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_177.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_178.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_179.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_180.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_181.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_182.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_183.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_184.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_185.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_186.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_187.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_188.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_189.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_190.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_191.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_192.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_193.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_194.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_195.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_196.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_197.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_198.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_199.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_200.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_201.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_202.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_203.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_204.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_205.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_206.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_207.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_208.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_209.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_210.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_211.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_212.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_213.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_214.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_215.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_216.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_217.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_218.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_219.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_220.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_221.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_222.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_223.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_224.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_225.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_226.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_227.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_228.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_229.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_230.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_231.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_232.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_233.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_234.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_235.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_236.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_237.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_238.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_239.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_240.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_241.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_242.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_243.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_244.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_245.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_246.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_247.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_248.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_249.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_250.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_251.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_252.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_253.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_254.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_255.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_256.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_257.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_258.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_259.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_260.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_261.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_262.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_263.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_264.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_265.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_266.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_267.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_268.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_269.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_270.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_271.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_272.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_273.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_274.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_275.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_276.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_277.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_278.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_279.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_280.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_281.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_282.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_283.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_284.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_285.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_286.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_287.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_288.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_289.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_290.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_291.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_292.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_293.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_294.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_295.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_296.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_297.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_298.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_299.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_300.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_301.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_302.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_303.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_304.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_305.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_306.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_307.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_308.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_309.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_310.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_311.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_312.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_313.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_314.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_315.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_316.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_317.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_318.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_319.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_320.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_321.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_322.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_323.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_324.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_325.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_326.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_327.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_328.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_329.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_330.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_331.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_332.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_333.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_334.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_335.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_336.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental//_337.sh

EOF