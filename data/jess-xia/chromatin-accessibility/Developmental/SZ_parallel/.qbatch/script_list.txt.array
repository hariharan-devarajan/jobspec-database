#!/bin/sh
#SBATCH --nodes=1
#SBATCH 
#SBATCH --output=/external/rprshnas01/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/logs/slurm-script_list.txt-%A_%a.out
#SBATCH --chdir=/external/rprshnas01/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel
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
export OLDPWD="/nethome/kcni/jxia/chromatin-accessibility/Developmental"
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
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_1.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_2.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_3.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_4.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_5.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_6.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_7.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_8.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_9.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_10.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_11.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_12.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_13.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_14.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_15.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_16.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_17.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_18.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_19.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_20.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_21.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_22.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_23.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_24.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_25.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_26.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_27.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_28.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_29.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_30.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_31.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_32.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_33.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_34.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_35.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_36.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_37.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_38.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_39.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_40.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_41.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_42.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_43.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_44.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_45.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_46.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_47.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_48.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_49.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_50.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_51.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_52.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_53.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_54.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_55.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_56.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_57.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_58.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_59.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_60.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_61.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_62.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_63.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_64.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_65.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_66.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_67.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_68.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_69.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_70.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_71.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_72.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_73.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_74.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_75.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_76.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_77.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_78.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_79.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_80.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_81.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_82.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_83.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_84.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_85.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_86.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_87.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_88.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_89.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_90.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_91.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_92.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_93.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_94.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_95.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_96.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_97.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_98.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_99.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_100.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_101.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_102.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_103.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_104.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_105.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_106.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_107.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_108.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_109.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_110.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_111.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_112.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_113.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_114.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_115.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_116.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_117.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_118.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_119.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_120.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_121.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_122.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_123.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_124.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_125.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_126.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_127.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_128.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_129.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_130.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_131.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_132.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_133.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_134.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_135.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_136.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_137.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_138.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_139.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_140.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_141.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_142.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_143.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_144.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_145.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_146.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_147.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_148.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_149.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_150.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_151.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_152.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_153.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_154.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_155.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_156.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_157.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_158.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_159.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_160.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_161.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_162.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_163.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_164.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_165.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_166.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_167.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_168.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_169.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_170.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_171.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_172.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_173.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_174.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_175.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_176.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_177.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_178.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_179.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_180.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_181.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_182.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_183.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_184.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_185.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_186.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_187.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_188.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_189.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_190.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_191.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_192.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_193.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_194.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_195.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_196.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_197.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_198.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_199.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_200.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_201.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_202.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_203.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_204.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_205.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_206.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_207.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_208.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_209.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_210.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_211.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_212.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_213.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_214.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_215.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_216.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_217.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_218.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_219.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_220.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_221.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_222.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_223.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_224.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_225.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_226.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_227.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_228.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_229.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_230.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_231.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_232.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_233.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_234.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_235.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_236.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_237.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_238.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_239.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_240.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_241.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_242.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_243.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_244.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_245.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_246.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_247.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_248.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_249.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_250.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_251.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_252.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_253.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_254.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_255.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_256.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_257.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_258.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_259.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_260.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_261.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_262.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_263.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_264.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_265.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_266.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_267.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_268.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_269.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_270.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_271.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_272.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_273.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_274.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_275.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_276.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_277.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_278.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_279.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_280.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_281.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_282.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_283.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_284.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_285.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_286.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_287.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_288.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_289.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_290.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_291.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_292.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_293.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_294.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_295.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_296.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_297.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_298.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_299.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_300.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_301.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_302.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_303.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_304.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_305.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_306.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_307.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_308.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_309.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_310.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_311.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_312.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_313.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_314.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_315.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_316.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_317.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_318.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_319.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_320.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_321.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_322.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_323.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_324.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_325.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_326.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_327.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_328.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_329.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_330.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_331.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_332.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_333.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_334.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_335.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_336.sh
/nethome/kcni/jxia/chromatin-accessibility/Developmental/SZ_parallel/SZ_337.sh

EOF