#!/bin/sh
#SBATCH --nodes=1
#SBATCH 
#SBATCH --output=/external/rprshnas01/kcni/jxia/chromatin-accessibility/logs/slurm-script_list.txt-%A_%a.out
#SBATCH --chdir=/external/rprshnas01/kcni/jxia/chromatin-accessibility
#SBATCH --job-name=script_list.txt
#SBATCH 
#SBATCH 
#SBATCH --array=1-342
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
export XDG_SESSION_ID="41348"
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
export DISPLAY="localhost:10.0"
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
export SSH_CONNECTION="172.28.201.186 50126 172.26.216.50 22"
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
export SSH_TTY="/dev/pts/3"
export OLDPWD="/nethome/kcni/jxia"
export LMOD_SYSTEM_DEFAULT_MODULES="SCC"
export LMOD_CACHED_LOADS="no"
export SSH_CLIENT="172.28.201.186 50126 22"
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
/nethome/kcni/jxia/chromatin-accessibility//_1.sh
/nethome/kcni/jxia/chromatin-accessibility//_2.sh
/nethome/kcni/jxia/chromatin-accessibility//_3.sh
/nethome/kcni/jxia/chromatin-accessibility//_4.sh
/nethome/kcni/jxia/chromatin-accessibility//_5.sh
/nethome/kcni/jxia/chromatin-accessibility//_6.sh
/nethome/kcni/jxia/chromatin-accessibility//_7.sh
/nethome/kcni/jxia/chromatin-accessibility//_8.sh
/nethome/kcni/jxia/chromatin-accessibility//_9.sh
/nethome/kcni/jxia/chromatin-accessibility//_10.sh
/nethome/kcni/jxia/chromatin-accessibility//_11.sh
/nethome/kcni/jxia/chromatin-accessibility//_12.sh
/nethome/kcni/jxia/chromatin-accessibility//_13.sh
/nethome/kcni/jxia/chromatin-accessibility//_14.sh
/nethome/kcni/jxia/chromatin-accessibility//_15.sh
/nethome/kcni/jxia/chromatin-accessibility//_16.sh
/nethome/kcni/jxia/chromatin-accessibility//_17.sh
/nethome/kcni/jxia/chromatin-accessibility//_18.sh
/nethome/kcni/jxia/chromatin-accessibility//_19.sh
/nethome/kcni/jxia/chromatin-accessibility//_20.sh
/nethome/kcni/jxia/chromatin-accessibility//_21.sh
/nethome/kcni/jxia/chromatin-accessibility//_22.sh
/nethome/kcni/jxia/chromatin-accessibility//_23.sh
/nethome/kcni/jxia/chromatin-accessibility//_24.sh
/nethome/kcni/jxia/chromatin-accessibility//_25.sh
/nethome/kcni/jxia/chromatin-accessibility//_26.sh
/nethome/kcni/jxia/chromatin-accessibility//_27.sh
/nethome/kcni/jxia/chromatin-accessibility//_28.sh
/nethome/kcni/jxia/chromatin-accessibility//_29.sh
/nethome/kcni/jxia/chromatin-accessibility//_30.sh
/nethome/kcni/jxia/chromatin-accessibility//_31.sh
/nethome/kcni/jxia/chromatin-accessibility//_32.sh
/nethome/kcni/jxia/chromatin-accessibility//_33.sh
/nethome/kcni/jxia/chromatin-accessibility//_34.sh
/nethome/kcni/jxia/chromatin-accessibility//_35.sh
/nethome/kcni/jxia/chromatin-accessibility//_36.sh
/nethome/kcni/jxia/chromatin-accessibility//_37.sh
/nethome/kcni/jxia/chromatin-accessibility//_38.sh
/nethome/kcni/jxia/chromatin-accessibility//_39.sh
/nethome/kcni/jxia/chromatin-accessibility//_40.sh
/nethome/kcni/jxia/chromatin-accessibility//_41.sh
/nethome/kcni/jxia/chromatin-accessibility//_42.sh
/nethome/kcni/jxia/chromatin-accessibility//_43.sh
/nethome/kcni/jxia/chromatin-accessibility//_44.sh
/nethome/kcni/jxia/chromatin-accessibility//_45.sh
/nethome/kcni/jxia/chromatin-accessibility//_46.sh
/nethome/kcni/jxia/chromatin-accessibility//_47.sh
/nethome/kcni/jxia/chromatin-accessibility//_48.sh
/nethome/kcni/jxia/chromatin-accessibility//_49.sh
/nethome/kcni/jxia/chromatin-accessibility//_50.sh
/nethome/kcni/jxia/chromatin-accessibility//_51.sh
/nethome/kcni/jxia/chromatin-accessibility//_52.sh
/nethome/kcni/jxia/chromatin-accessibility//_53.sh
/nethome/kcni/jxia/chromatin-accessibility//_54.sh
/nethome/kcni/jxia/chromatin-accessibility//_55.sh
/nethome/kcni/jxia/chromatin-accessibility//_56.sh
/nethome/kcni/jxia/chromatin-accessibility//_57.sh
/nethome/kcni/jxia/chromatin-accessibility//_58.sh
/nethome/kcni/jxia/chromatin-accessibility//_59.sh
/nethome/kcni/jxia/chromatin-accessibility//_60.sh
/nethome/kcni/jxia/chromatin-accessibility//_61.sh
/nethome/kcni/jxia/chromatin-accessibility//_62.sh
/nethome/kcni/jxia/chromatin-accessibility//_63.sh
/nethome/kcni/jxia/chromatin-accessibility//_64.sh
/nethome/kcni/jxia/chromatin-accessibility//_65.sh
/nethome/kcni/jxia/chromatin-accessibility//_66.sh
/nethome/kcni/jxia/chromatin-accessibility//_67.sh
/nethome/kcni/jxia/chromatin-accessibility//_68.sh
/nethome/kcni/jxia/chromatin-accessibility//_69.sh
/nethome/kcni/jxia/chromatin-accessibility//_70.sh
/nethome/kcni/jxia/chromatin-accessibility//_71.sh
/nethome/kcni/jxia/chromatin-accessibility//_72.sh
/nethome/kcni/jxia/chromatin-accessibility//_73.sh
/nethome/kcni/jxia/chromatin-accessibility//_74.sh
/nethome/kcni/jxia/chromatin-accessibility//_75.sh
/nethome/kcni/jxia/chromatin-accessibility//_76.sh
/nethome/kcni/jxia/chromatin-accessibility//_77.sh
/nethome/kcni/jxia/chromatin-accessibility//_78.sh
/nethome/kcni/jxia/chromatin-accessibility//_79.sh
/nethome/kcni/jxia/chromatin-accessibility//_80.sh
/nethome/kcni/jxia/chromatin-accessibility//_81.sh
/nethome/kcni/jxia/chromatin-accessibility//_82.sh
/nethome/kcni/jxia/chromatin-accessibility//_83.sh
/nethome/kcni/jxia/chromatin-accessibility//_84.sh
/nethome/kcni/jxia/chromatin-accessibility//_85.sh
/nethome/kcni/jxia/chromatin-accessibility//_86.sh
/nethome/kcni/jxia/chromatin-accessibility//_87.sh
/nethome/kcni/jxia/chromatin-accessibility//_88.sh
/nethome/kcni/jxia/chromatin-accessibility//_89.sh
/nethome/kcni/jxia/chromatin-accessibility//_90.sh
/nethome/kcni/jxia/chromatin-accessibility//_91.sh
/nethome/kcni/jxia/chromatin-accessibility//_92.sh
/nethome/kcni/jxia/chromatin-accessibility//_93.sh
/nethome/kcni/jxia/chromatin-accessibility//_94.sh
/nethome/kcni/jxia/chromatin-accessibility//_95.sh
/nethome/kcni/jxia/chromatin-accessibility//_96.sh
/nethome/kcni/jxia/chromatin-accessibility//_97.sh
/nethome/kcni/jxia/chromatin-accessibility//_98.sh
/nethome/kcni/jxia/chromatin-accessibility//_99.sh
/nethome/kcni/jxia/chromatin-accessibility//_100.sh
/nethome/kcni/jxia/chromatin-accessibility//_101.sh
/nethome/kcni/jxia/chromatin-accessibility//_102.sh
/nethome/kcni/jxia/chromatin-accessibility//_103.sh
/nethome/kcni/jxia/chromatin-accessibility//_104.sh
/nethome/kcni/jxia/chromatin-accessibility//_105.sh
/nethome/kcni/jxia/chromatin-accessibility//_106.sh
/nethome/kcni/jxia/chromatin-accessibility//_107.sh
/nethome/kcni/jxia/chromatin-accessibility//_108.sh
/nethome/kcni/jxia/chromatin-accessibility//_109.sh
/nethome/kcni/jxia/chromatin-accessibility//_110.sh
/nethome/kcni/jxia/chromatin-accessibility//_111.sh
/nethome/kcni/jxia/chromatin-accessibility//_112.sh
/nethome/kcni/jxia/chromatin-accessibility//_113.sh
/nethome/kcni/jxia/chromatin-accessibility//_114.sh
/nethome/kcni/jxia/chromatin-accessibility//_115.sh
/nethome/kcni/jxia/chromatin-accessibility//_116.sh
/nethome/kcni/jxia/chromatin-accessibility//_117.sh
/nethome/kcni/jxia/chromatin-accessibility//_118.sh
/nethome/kcni/jxia/chromatin-accessibility//_119.sh
/nethome/kcni/jxia/chromatin-accessibility//_120.sh
/nethome/kcni/jxia/chromatin-accessibility//_121.sh
/nethome/kcni/jxia/chromatin-accessibility//_122.sh
/nethome/kcni/jxia/chromatin-accessibility//_123.sh
/nethome/kcni/jxia/chromatin-accessibility//_124.sh
/nethome/kcni/jxia/chromatin-accessibility//_125.sh
/nethome/kcni/jxia/chromatin-accessibility//_126.sh
/nethome/kcni/jxia/chromatin-accessibility//_127.sh
/nethome/kcni/jxia/chromatin-accessibility//_128.sh
/nethome/kcni/jxia/chromatin-accessibility//_129.sh
/nethome/kcni/jxia/chromatin-accessibility//_130.sh
/nethome/kcni/jxia/chromatin-accessibility//_131.sh
/nethome/kcni/jxia/chromatin-accessibility//_132.sh
/nethome/kcni/jxia/chromatin-accessibility//_133.sh
/nethome/kcni/jxia/chromatin-accessibility//_134.sh
/nethome/kcni/jxia/chromatin-accessibility//_135.sh
/nethome/kcni/jxia/chromatin-accessibility//_136.sh
/nethome/kcni/jxia/chromatin-accessibility//_137.sh
/nethome/kcni/jxia/chromatin-accessibility//_138.sh
/nethome/kcni/jxia/chromatin-accessibility//_139.sh
/nethome/kcni/jxia/chromatin-accessibility//_140.sh
/nethome/kcni/jxia/chromatin-accessibility//_141.sh
/nethome/kcni/jxia/chromatin-accessibility//_142.sh
/nethome/kcni/jxia/chromatin-accessibility//_143.sh
/nethome/kcni/jxia/chromatin-accessibility//_144.sh
/nethome/kcni/jxia/chromatin-accessibility//_145.sh
/nethome/kcni/jxia/chromatin-accessibility//_146.sh
/nethome/kcni/jxia/chromatin-accessibility//_147.sh
/nethome/kcni/jxia/chromatin-accessibility//_148.sh
/nethome/kcni/jxia/chromatin-accessibility//_149.sh
/nethome/kcni/jxia/chromatin-accessibility//_150.sh
/nethome/kcni/jxia/chromatin-accessibility//_151.sh
/nethome/kcni/jxia/chromatin-accessibility//_152.sh
/nethome/kcni/jxia/chromatin-accessibility//_153.sh
/nethome/kcni/jxia/chromatin-accessibility//_154.sh
/nethome/kcni/jxia/chromatin-accessibility//_155.sh
/nethome/kcni/jxia/chromatin-accessibility//_156.sh
/nethome/kcni/jxia/chromatin-accessibility//_157.sh
/nethome/kcni/jxia/chromatin-accessibility//_158.sh
/nethome/kcni/jxia/chromatin-accessibility//_159.sh
/nethome/kcni/jxia/chromatin-accessibility//_160.sh
/nethome/kcni/jxia/chromatin-accessibility//_161.sh
/nethome/kcni/jxia/chromatin-accessibility//_162.sh
/nethome/kcni/jxia/chromatin-accessibility//_163.sh
/nethome/kcni/jxia/chromatin-accessibility//_164.sh
/nethome/kcni/jxia/chromatin-accessibility//_165.sh
/nethome/kcni/jxia/chromatin-accessibility//_166.sh
/nethome/kcni/jxia/chromatin-accessibility//_167.sh
/nethome/kcni/jxia/chromatin-accessibility//_168.sh
/nethome/kcni/jxia/chromatin-accessibility//_169.sh
/nethome/kcni/jxia/chromatin-accessibility//_170.sh
/nethome/kcni/jxia/chromatin-accessibility//_171.sh
/nethome/kcni/jxia/chromatin-accessibility//_172.sh
/nethome/kcni/jxia/chromatin-accessibility//_173.sh
/nethome/kcni/jxia/chromatin-accessibility//_174.sh
/nethome/kcni/jxia/chromatin-accessibility//_175.sh
/nethome/kcni/jxia/chromatin-accessibility//_176.sh
/nethome/kcni/jxia/chromatin-accessibility//_177.sh
/nethome/kcni/jxia/chromatin-accessibility//_178.sh
/nethome/kcni/jxia/chromatin-accessibility//_179.sh
/nethome/kcni/jxia/chromatin-accessibility//_180.sh
/nethome/kcni/jxia/chromatin-accessibility//_181.sh
/nethome/kcni/jxia/chromatin-accessibility//_182.sh
/nethome/kcni/jxia/chromatin-accessibility//_183.sh
/nethome/kcni/jxia/chromatin-accessibility//_184.sh
/nethome/kcni/jxia/chromatin-accessibility//_185.sh
/nethome/kcni/jxia/chromatin-accessibility//_186.sh
/nethome/kcni/jxia/chromatin-accessibility//_187.sh
/nethome/kcni/jxia/chromatin-accessibility//_188.sh
/nethome/kcni/jxia/chromatin-accessibility//_189.sh
/nethome/kcni/jxia/chromatin-accessibility//_190.sh
/nethome/kcni/jxia/chromatin-accessibility//_191.sh
/nethome/kcni/jxia/chromatin-accessibility//_192.sh
/nethome/kcni/jxia/chromatin-accessibility//_193.sh
/nethome/kcni/jxia/chromatin-accessibility//_194.sh
/nethome/kcni/jxia/chromatin-accessibility//_195.sh
/nethome/kcni/jxia/chromatin-accessibility//_196.sh
/nethome/kcni/jxia/chromatin-accessibility//_197.sh
/nethome/kcni/jxia/chromatin-accessibility//_198.sh
/nethome/kcni/jxia/chromatin-accessibility//_199.sh
/nethome/kcni/jxia/chromatin-accessibility//_200.sh
/nethome/kcni/jxia/chromatin-accessibility//_201.sh
/nethome/kcni/jxia/chromatin-accessibility//_202.sh
/nethome/kcni/jxia/chromatin-accessibility//_203.sh
/nethome/kcni/jxia/chromatin-accessibility//_204.sh
/nethome/kcni/jxia/chromatin-accessibility//_205.sh
/nethome/kcni/jxia/chromatin-accessibility//_206.sh
/nethome/kcni/jxia/chromatin-accessibility//_207.sh
/nethome/kcni/jxia/chromatin-accessibility//_208.sh
/nethome/kcni/jxia/chromatin-accessibility//_209.sh
/nethome/kcni/jxia/chromatin-accessibility//_210.sh
/nethome/kcni/jxia/chromatin-accessibility//_211.sh
/nethome/kcni/jxia/chromatin-accessibility//_212.sh
/nethome/kcni/jxia/chromatin-accessibility//_213.sh
/nethome/kcni/jxia/chromatin-accessibility//_214.sh
/nethome/kcni/jxia/chromatin-accessibility//_215.sh
/nethome/kcni/jxia/chromatin-accessibility//_216.sh
/nethome/kcni/jxia/chromatin-accessibility//_217.sh
/nethome/kcni/jxia/chromatin-accessibility//_218.sh
/nethome/kcni/jxia/chromatin-accessibility//_219.sh
/nethome/kcni/jxia/chromatin-accessibility//_220.sh
/nethome/kcni/jxia/chromatin-accessibility//_221.sh
/nethome/kcni/jxia/chromatin-accessibility//_222.sh
/nethome/kcni/jxia/chromatin-accessibility//_223.sh
/nethome/kcni/jxia/chromatin-accessibility//_224.sh
/nethome/kcni/jxia/chromatin-accessibility//_225.sh
/nethome/kcni/jxia/chromatin-accessibility//_226.sh
/nethome/kcni/jxia/chromatin-accessibility//_227.sh
/nethome/kcni/jxia/chromatin-accessibility//_228.sh
/nethome/kcni/jxia/chromatin-accessibility//_229.sh
/nethome/kcni/jxia/chromatin-accessibility//_230.sh
/nethome/kcni/jxia/chromatin-accessibility//_231.sh
/nethome/kcni/jxia/chromatin-accessibility//_232.sh
/nethome/kcni/jxia/chromatin-accessibility//_233.sh
/nethome/kcni/jxia/chromatin-accessibility//_234.sh
/nethome/kcni/jxia/chromatin-accessibility//_235.sh
/nethome/kcni/jxia/chromatin-accessibility//_236.sh
/nethome/kcni/jxia/chromatin-accessibility//_237.sh
/nethome/kcni/jxia/chromatin-accessibility//_238.sh
/nethome/kcni/jxia/chromatin-accessibility//_239.sh
/nethome/kcni/jxia/chromatin-accessibility//_240.sh
/nethome/kcni/jxia/chromatin-accessibility//_241.sh
/nethome/kcni/jxia/chromatin-accessibility//_242.sh
/nethome/kcni/jxia/chromatin-accessibility//_243.sh
/nethome/kcni/jxia/chromatin-accessibility//_244.sh
/nethome/kcni/jxia/chromatin-accessibility//_245.sh
/nethome/kcni/jxia/chromatin-accessibility//_246.sh
/nethome/kcni/jxia/chromatin-accessibility//_247.sh
/nethome/kcni/jxia/chromatin-accessibility//_248.sh
/nethome/kcni/jxia/chromatin-accessibility//_249.sh
/nethome/kcni/jxia/chromatin-accessibility//_250.sh
/nethome/kcni/jxia/chromatin-accessibility//_251.sh
/nethome/kcni/jxia/chromatin-accessibility//_252.sh
/nethome/kcni/jxia/chromatin-accessibility//_253.sh
/nethome/kcni/jxia/chromatin-accessibility//_254.sh
/nethome/kcni/jxia/chromatin-accessibility//_255.sh
/nethome/kcni/jxia/chromatin-accessibility//_256.sh
/nethome/kcni/jxia/chromatin-accessibility//_257.sh
/nethome/kcni/jxia/chromatin-accessibility//_258.sh
/nethome/kcni/jxia/chromatin-accessibility//_259.sh
/nethome/kcni/jxia/chromatin-accessibility//_260.sh
/nethome/kcni/jxia/chromatin-accessibility//_261.sh
/nethome/kcni/jxia/chromatin-accessibility//_262.sh
/nethome/kcni/jxia/chromatin-accessibility//_263.sh
/nethome/kcni/jxia/chromatin-accessibility//_264.sh
/nethome/kcni/jxia/chromatin-accessibility//_265.sh
/nethome/kcni/jxia/chromatin-accessibility//_266.sh
/nethome/kcni/jxia/chromatin-accessibility//_267.sh
/nethome/kcni/jxia/chromatin-accessibility//_268.sh
/nethome/kcni/jxia/chromatin-accessibility//_269.sh
/nethome/kcni/jxia/chromatin-accessibility//_270.sh
/nethome/kcni/jxia/chromatin-accessibility//_271.sh
/nethome/kcni/jxia/chromatin-accessibility//_272.sh
/nethome/kcni/jxia/chromatin-accessibility//_273.sh
/nethome/kcni/jxia/chromatin-accessibility//_274.sh
/nethome/kcni/jxia/chromatin-accessibility//_275.sh
/nethome/kcni/jxia/chromatin-accessibility//_276.sh
/nethome/kcni/jxia/chromatin-accessibility//_277.sh
/nethome/kcni/jxia/chromatin-accessibility//_278.sh
/nethome/kcni/jxia/chromatin-accessibility//_279.sh
/nethome/kcni/jxia/chromatin-accessibility//_280.sh
/nethome/kcni/jxia/chromatin-accessibility//_281.sh
/nethome/kcni/jxia/chromatin-accessibility//_282.sh
/nethome/kcni/jxia/chromatin-accessibility//_283.sh
/nethome/kcni/jxia/chromatin-accessibility//_284.sh
/nethome/kcni/jxia/chromatin-accessibility//_285.sh
/nethome/kcni/jxia/chromatin-accessibility//_286.sh
/nethome/kcni/jxia/chromatin-accessibility//_287.sh
/nethome/kcni/jxia/chromatin-accessibility//_288.sh
/nethome/kcni/jxia/chromatin-accessibility//_289.sh
/nethome/kcni/jxia/chromatin-accessibility//_290.sh
/nethome/kcni/jxia/chromatin-accessibility//_291.sh
/nethome/kcni/jxia/chromatin-accessibility//_292.sh
/nethome/kcni/jxia/chromatin-accessibility//_293.sh
/nethome/kcni/jxia/chromatin-accessibility//_294.sh
/nethome/kcni/jxia/chromatin-accessibility//_295.sh
/nethome/kcni/jxia/chromatin-accessibility//_296.sh
/nethome/kcni/jxia/chromatin-accessibility//_297.sh
/nethome/kcni/jxia/chromatin-accessibility//_298.sh
/nethome/kcni/jxia/chromatin-accessibility//_299.sh
/nethome/kcni/jxia/chromatin-accessibility//_300.sh
/nethome/kcni/jxia/chromatin-accessibility//_301.sh
/nethome/kcni/jxia/chromatin-accessibility//_302.sh
/nethome/kcni/jxia/chromatin-accessibility//_303.sh
/nethome/kcni/jxia/chromatin-accessibility//_304.sh
/nethome/kcni/jxia/chromatin-accessibility//_305.sh
/nethome/kcni/jxia/chromatin-accessibility//_306.sh
/nethome/kcni/jxia/chromatin-accessibility//_307.sh
/nethome/kcni/jxia/chromatin-accessibility//_308.sh
/nethome/kcni/jxia/chromatin-accessibility//_309.sh
/nethome/kcni/jxia/chromatin-accessibility//_310.sh
/nethome/kcni/jxia/chromatin-accessibility//_311.sh
/nethome/kcni/jxia/chromatin-accessibility//_312.sh
/nethome/kcni/jxia/chromatin-accessibility//_313.sh
/nethome/kcni/jxia/chromatin-accessibility//_314.sh
/nethome/kcni/jxia/chromatin-accessibility//_315.sh
/nethome/kcni/jxia/chromatin-accessibility//_316.sh
/nethome/kcni/jxia/chromatin-accessibility//_317.sh
/nethome/kcni/jxia/chromatin-accessibility//_318.sh
/nethome/kcni/jxia/chromatin-accessibility//_319.sh
/nethome/kcni/jxia/chromatin-accessibility//_320.sh
/nethome/kcni/jxia/chromatin-accessibility//_321.sh
/nethome/kcni/jxia/chromatin-accessibility//_322.sh
/nethome/kcni/jxia/chromatin-accessibility//_323.sh
/nethome/kcni/jxia/chromatin-accessibility//_324.sh
/nethome/kcni/jxia/chromatin-accessibility//_325.sh
/nethome/kcni/jxia/chromatin-accessibility//_326.sh
/nethome/kcni/jxia/chromatin-accessibility//_327.sh
/nethome/kcni/jxia/chromatin-accessibility//_328.sh
/nethome/kcni/jxia/chromatin-accessibility//_329.sh
/nethome/kcni/jxia/chromatin-accessibility//_330.sh
/nethome/kcni/jxia/chromatin-accessibility//_331.sh
/nethome/kcni/jxia/chromatin-accessibility//_332.sh
/nethome/kcni/jxia/chromatin-accessibility//_333.sh
/nethome/kcni/jxia/chromatin-accessibility//_334.sh
/nethome/kcni/jxia/chromatin-accessibility//_335.sh
/nethome/kcni/jxia/chromatin-accessibility//_336.sh
/nethome/kcni/jxia/chromatin-accessibility//_337.sh
/nethome/kcni/jxia/chromatin-accessibility//_338.sh
/nethome/kcni/jxia/chromatin-accessibility//_339.sh
/nethome/kcni/jxia/chromatin-accessibility//_340.sh
/nethome/kcni/jxia/chromatin-accessibility//_341.sh
/nethome/kcni/jxia/chromatin-accessibility//_342.sh

EOF