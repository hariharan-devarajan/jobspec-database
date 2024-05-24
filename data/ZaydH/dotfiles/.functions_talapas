#!/usr/bin/env bash

# Kills a slurm job
function kill_job {
    DRY_RUN=0
    unset JOB_NUM  # Ensure
    JOB_DESC="${0} [-d]"
    while ! [ -z $1 ] ; do
        case "$1" in
            -d)  # Dry run
                DRY_RUN=1
                # echo "Running in dry run mode"
                ;;
            *)
                if [ -z "${JOB_NUM}" ] ; then
                    JOB_NUM="$1"
                else
                    echo "Too many input arguments. Run as '${JOB_DESC} jobnum"
                    return 1
                fi
                ;;
        esac
        shift
    done

    if ! $(is_number "${JOB_NUM}"); then
        echo "Specified job number \"${JOB_NUM}\" is not a number. Exiting..."
        return 1
    fi
    # Based on: https://stackoverflow.com/questions/52263932/slurm-how-to-obtain-only-jobid-using-jobname-through-a-script
    #          -n will suppress the header
    #          -X will suppress the .batch part
    JOB_NAME=$( squeue --noheader -u $(whoami) --format='%j'  -j "${JOB_NUM}" )
    JOB_PARTITION=$( squeue --noheader -u $(whoami) --format='%P'  -j "${JOB_NUM}" )
    JOB_STATUS=$( squeue --noheader -u $(whoami) --format='%T'  -j "${JOB_NUM}" )
    # for job_info in $(squeue --user="$myself" --noheader --format='%i;%j') ; do
    if [ -z "${JOB_NAME}" ]; then
        echo "Job number ${JOB_NUM} does not appear to be an active job (or one you own). Check your job number and try again. Exiting..."
        return 1
    fi


    job_desc="${JOB_NUM} (${JOB_NAME}) -- Partition: ${JOB_PARTITION}, Status: ${JOB_STATUS}"
    if [ ${DRY_RUN} -eq 0 ]; then
        echo "Canceling Job: ${job_desc}"
        scancel ${JOB_NUM}
        if [ $? -ne 0 ] ; then
            echo "Job cancel appears to have failed. Check the job status to be sure."
        fi
    else
        echo "Dry run cancel: ${job_desc}"
    fi
}

# Returns true if the input argument is a (positive) integer
function is_number {
    if [ -z "$1" ]; then
        printf "No input string specified\n"
        return 1
    elif (( $# > 1 )); then
        printf "Invalid argument count\n"
        return 1
    fi
    re='^[0-9]+$'
    [[ "$1" =~ "${re}" ]]
}

# Function that kills all jobs whose ID is greater than or equal to the specified parameter
function kill_jobs_larger {
    if [ $# -lt 1 ]; then
        echo "At least one argument required"
        return
    fi

    local -a jobs
    DRY_RUN=0
    JOB_DESC="${0} [-d]"
    unset minjobnum # Ensure
    while ! [ -z $1 ] ; do
        case "$1" in
            -d)  # Dry run
                DRY_RUN=1
                echo "Running in dry run mode"
                ;;
            *)
                if [ -z "${minjobnum}" ] ; then
                    minjobnum="$1"
                else
                    echo "Too many input arguments. Run as '${JOB_DESC} minjobnum'"
                    return 1
                fi
                ;;
        esac
        shift
    done

    myself="$(id -u -n)"

    local -a jobs  # Array of the jobnumbers to cancel
    for job_info in $(squeue --user="$myself" --noheader --format='%i;%j;%P;%T') ; do
      j="$( cut -d ';' -f 1 <<< "$job_info" )"
      if [ "$j" -ge "${minjobnum}" ] ; then
        name="$( cut -d ';' -f 2 <<< "$job_info" )"
        partition="$( cut -d ';' -f 3 <<< "$job_info" )"
        job_status="$( cut -d ';' -f 4 <<< "$job_info" )"

        job_desc="$j (${name}) -- Partition: ${partition}, Status: ${job_status}"
        if [ ${DRY_RUN} -eq 0 ]; then
            jobs+=($j)
            printf "Canceling Job: ${job_desc}\n"
        else
            printf "Dry run cancel: ${job_desc}\n"
        fi
      fi
    done

    # Only cancel if jobs list isn't empty
    if [ ${#jobs[@]} -gt 0 ]; then
        scancel "${jobs[@]}"
    fi
}

# Function that kills all jobs whose ID within the specified range (inclusive)
function kill_jobs_range {
    if [ $# -lt 2 ]; then
        echo "At least two arguments required"
        return
    fi

    DRY_RUN=0
    JOB_DESC="${0} [-d]"
    unset minjobnum maxjobnum  # Ensure
    while ! [ -z $1 ] ; do
        case "$1" in
            -d)  # Dry run
                DRY_RUN=1
                echo "Running in dry run mode"
                ;;
            *)
                if [ -z "${minjobnum}" ] ; then
                    minjobnum="$1"
                elif [ -z "${maxjobnum}" ]; then
                    maxjobnum="$1"
                else
                    echo "Too many input arguments. Run as '${JOB_DESC} minjobnum maxjobnum'"
                    return 1
                fi
                ;;
        esac
        shift
    done

    # Error check
    if [ -z "${minjobnum}" ]; then
        printf "Minimum Job Number argument is required. Run as '${JOB_DESC} minjobnum maxjobnum'\n"
        return 1
    elif ! $(is_number "$minjobnum"); then
        printf "Input argument 'minjobnum' is not an integer\n"
        return 1
    elif [ -z "${maxjobnum}" ] ; then
        printf "Maximum Job Number argument is required. Run as '${JOB_DESC} $1 maxjobnum'\n"
        return 1
    elif ! $(is_number "${maxjobnum}"); then
        printf "Input argument 'maxjobnum' is not an integer\n"
        return 1
    fi

    myself="$(id -u -n)"

    local -a jobs  # Array of the jobnumbers to cancel
    for job_info in $(squeue --user="$myself" --noheader --format='%i;%j;%P;%T') ; do
      j="$( cut -d ';' -f 1 <<< "$job_info" )"
      name="$( cut -d ';' -f 2 <<< "$job_info" )"
      partition="$( cut -d ';' -f 3 <<< "$job_info" )"
      job_status="$( cut -d ';' -f 4 <<< "$job_info" )"

      job_desc="$j (${name}) -- Partition: ${partition}, Status: ${job_status}"
      if [ "$j" -ge "${minjobnum}" -a "$j" -le "${maxjobnum}" ] ; then
        if [ ${DRY_RUN} -eq 0 ]; then
            jobs+=($j)
            printf "Canceling Job: ${job_desc}\n"
        else
            printf "Dry run cancel: ${job_desc}\n"
        fi
      fi
    done

    # Only cancel if jobs list isn't empty
    if [ ${#jobs[@]} -gt 0 ]; then
        scancel "${jobs[@]}"
    fi
}

# Runs a command line function call automatically in a slurm job
function run_cli_slurm() {
    if [ $# -ne 3 ]; then
        printf "Invalid argument count.\nExpected format: run_cli_slurm <JobName> <Partition> <CLI_COMMAND>\n"
        return 0
    fi
    JOB_NAME=$1
    TMP_PARTITION=$2  # Cannot name PARTITION as may have scope collision
    CLI_COMMAND=$3

    OUT_DIR=/home/zhammoud/projects/.out
    mkdir -p ${OUT_DIR}

    REQUEUE_SUFFIX="-rq"
    CPU_SUFFIX="-cpu"

    DEFAULT_GPU_PARTITION="gpu"
    A100_GPU_PARTITION="a100gpu"
    TEST_PARTITION="test"
    LONG_PARTITION="long"
    SHORT_PARTITION="short"
    PREEMPT_PARTITION="preempt"
    LOWD_PARTITION="lowd"

    if [[ ${TMP_PARTITION} = ${SHORT_PARTITION} || ${TMP_PARTITION} = ${DEFAULT_GPU_PARTITION} ]]; then
        TIME_LIMIT="1-00:00:00"  # 1 day
    elif [[ ${TMP_PARTITION} == "${TEST_PARTITION}"* ]]; then
        TIME_LIMIT="4:00:00"  # 4 hours
    elif [[ ${TMP_PARTITION} == "${LONG_PARTITION}"* ]]; then
        TIME_LIMIT="14-00:00:00"  # 14 day
    elif [[ ${TMP_PARTITION} == "${PREEMPT_PARTITION}"* ]]; then
        TIME_LIMIT="07-00:00:00"  # 7 day
    elif [[ ${TMP_PARTITION} == "${A100_GPU_PARTITION}"* ]]; then
        TIME_LIMIT="01-00:00:00"  # 1 day
    elif [[ ${TMP_PARTITION} == "${LOWD_PARTITION}"* ]]; then
        TIME_LIMIT="14-00:00:00"  # 1 day
    fi

    REQUEUE=false
    if [[ ${TMP_PARTITION} == *"${REQUEUE_SUFFIX}" ]] ; then
        TMP_PARTITION="${TMP_PARTITION%$REQUEUE_SUFFIX}"
        REQUEUE=true
    fi

    N_CPU=3
    N_GPU=0
    RAM_PER_CPU=16G
    if [[ ${TMP_PARTITION} == *"testgpu"* ]]; then
        N_CPU=2
        N_GPU=1
    elif [[ ${TMP_PARTITION} == "short" ]]; then
        N_CPU=1
        N_GPU=0
        RAM_PER_CPU=40G
    elif [[ ${TMP_PARTITION} == *"${CPU_SUFFIX}" ]] ; then
        N_CPU=6
        N_GPU=0
        TMP_PARTITION="${TMP_PARTITION%$CPU_SUFFIX}"  # Strip the suffix
        RAM_PER_CPU=16G
    # elif [[ ${TMP_PARTITION} =~ "\(.*gpu\|${PREEMPT_PARTITION}.*\|${LOWD_PARTITION}\)" ]] ; then
    elif [[ "${TMP_PARTITION}" =~ (${A100_GPU_PARTITION}|${PREEMPT_PARTITION}.*|${LOWD_PARTITION}) ]] ; then
        N_GPU=1
    fi

    SBATCH_TEXT=$(echo "#!/bin/bash" \
                       "\n#SBATCH --job-name=${JOB_NAME}" \
                       "\n#SBATCH --account=uoml" \
                       "\n#SBATCH --partition=${TMP_PARTITION}" \
                       "\n#SBATCH --time=${TIME_LIMIT}" \
                       "\n#SBATCH --mem-per-cpu=${RAM_PER_CPU}" \
                       "\n#SBATCH --nodes=1"\
                       "\n#SBATCH --cpus-per-task=${N_CPU}" \
                       "\n#SBATCH --ntasks-per-node=1" \
                       "\n#SBATCH --mail-type=ALL" \
                       "\n#SBATCH --mail-user=zhammoud@uoregon.edu" \
                       "\n#SBATCH --output=${OUT_DIR}/${JOB_NAME}_%j.out" \
                 )

    # Select to use GPU. Must be before package loading.
    if [[ "${N_GPU}" -gt "0" ]] ; then
        SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --gres=gpu:${N_GPU}"
        if [[ ${TMP_PARTITION} == "${PREEMPT_PARTITION}" ]]; then
            :  # Allows empty if blocks
            # SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --constraint=a100"  # Forces to use A100
            # SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --constraint=v100"   # Forces to use V100
        elif [[ "${PARTITION}" == "${A100_GPU_PARTITION}" ]]; then
            SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --constraint=a100"    # A100 partition only has A100s
            SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --constraint=gpu-40gb"
        elif [[ ${TMP_PARTITION} == "${LOWD_PARTITION}" ]]; then
            SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --constraint=a100"  # Forces to use A100
        fi
    fi

    # Optionally requeue
    if ${REQUEUE} ; then
        SBATCH_TEXT="${SBATCH_TEXT}\n#SBATCH --requeue"
    fi

    SBATCH_TEXT=$(echo "${SBATCH_TEXT}" \
                  "\n" \
                  "\nmodule load racs-spack" \
                  "\nmodule load cuda/11.5.1" \
                  "\nmodule load cmake/3.15.4" \
                  "\nmodule load gurobi" \
                  "\n" \
                  "\nspack load gcc@8.2.0" \
                  "\n" \
                  "\n# Variables specifically for PyEnv" \
                  "\nexport PYENV_ROOT=\"\${HOME}/.pyenv\"" \
                  "\nif [ -d \"\${PYENV_ROOT}\" ]; then" \
                  "\n    export PATH=\"\${PYENV_ROOT}/bin:\${PATH}\"" \
                  "\n    eval \"\$(pyenv init -)\"" \
                  "\nfi" \
                 )
    if [[ "${N_GPU}" -gt "0" ]] ; then
        SBATCH_TEXT="${SBATCH_TEXT}\n\nif [[ \$(nvidia-smi) == *\"Tesla K80\"* ]] ; then\n echo \"PyEnv Local: 3.7.13\" \n pyenv shell 3.7.13\nfi"
    fi

    # Combine the user specified command at the end to ensure all modules
    # are successfully loaded.
    SBATCH_TEXT="${SBATCH_TEXT}\n\n${CLI_COMMAND}\n"
    echo -en "${SBATCH_TEXT}" > cli.slurm
    echo -en "${SBATCH_TEXT}" | sbatch /dev/stdin
}

function run_talapas_bash() {
    if [[ $# -eq 0 ]]; then
        PARTITION="testgpu"
        printf "No partition specified. Using partition \"${PARTITION}\"\n"
    elif [[ $# -eq 1 ]]; then
        PARTITION=$1
        printf "Specified partition: \"${PARTITION}\"\n"
    else
        printf "Invalid command input. Valid call \"run_talapas_bash [PARTITION]\"...\n"
        return 1
    fi
    if [[ "${PARTITION}" == "testgpu" ]]; then
        NUM_CPU=3
    elif [[ "${PARTITION}" == "short" ]]; then
        NUM_CPU=1
        srun --account=uoml --partition="${PARTITION}" --mem-per-cpu=28G --cpus-per-task=${NUM_CPU} --pty bash
        return
    else
        NUM_CPU=3
    fi
    printf "Number of CPUs Allocated: ${NUM_CPU}\n"

    SRUN_PARAMS=(
        "--account=uoml"
        "--partition=${PARTITION}"
        "--mem-per-cpu=16G"
        "--cpus-per-task=${NUM_CPU}"
        "--gres=gpu:1"
    )
    if [[ "${PARTITION}" == "preempt" ]]; then
        :  # Allows empty if blocks
        # SRUN_PARAMS+=("--constraint=a100")  # Forces to use A100
        # SRUN_PARAMS+=("--constraint=v100")  # Forces to use V100
    elif [[ "${PARTITION}" == "a100gpu" ]]; then
        SRUN_PARAMS+=("--constraint=a100")  # Partition only has A100s
        SRUN_PARAMS+=("--constraint=gpu-40gb")
    elif [[ "${PARTITION}" == "lowd" ]]; then
        SRUN_PARAMS+=("--constraint=a100")  # Forces to use A100
        SRUN_PARAMS+=("--constraint=gpu-80gb")
    fi

    echo "${SRUN_PARAMS[@]}"
    srun "${SRUN_PARAMS[@]}" --pty bash
    # srun --constraint="ampere" --account=uoml --partition="${PARTITION}" --mem-per-cpu=28G --gres=gpu:1 --cpus-per-task=${NUM_CPU} --pty bash
}

# Returns true if current host if Talapas node is a K80
function is_k80() {
    [[ $( nvidia-smi ) == *"Tesla K80"* ]]
    # return implicit and not needed
}
