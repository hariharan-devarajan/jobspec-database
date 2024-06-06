#!/bin/bash

#BSUB -G asccasc
#BSUB -q pbatch
#BSUB -nnodes 32
#BSUB -W 720
echo $LLNL_NUM_COMPUTE_NODES

min_num_nodes=$LLNL_NUM_COMPUTE_NODES
total_num_nodes=$LLNL_NUM_COMPUTE_NODES
tasks_per_node=16

echo $USER

generate_file_list()
{
    local file_prefix=$1
    local start_fileno=$2
    local num_of_files=$3
    local end_fileno=$((start_fileno + num_of_files - 1))
    local fileno=$start_fileno
    local filelist=
    while [ $fileno -le $end_fileno ]
    do
        filelist=$filelist$file_prefix$fileno
        if [ $fileno -ne $end_fileno ]
        then
            filelist=$filelist:
        fi
        fileno=$((fileno+1))
    done
    echo $filelist
}

generate_uniform_elem_list()
{
    local elem_val=$1
    local num_of_elems=$2
    local elemno=1
    local elemlist=
    while [ $elemno -le $num_of_elems ]
    do
        elemlist=$elemlist$elem_val
        if [ $elemno -ne $num_of_elems ]
        then
            elemlist=$elemlist:
        fi
        elemno=$((elemno+1))
    done
    echo $elemlist
}

create_delim_separated_task_list()
{
    local list_size=$1
    local delim=$2
    local taskid=0
    local task_list=
    while [ $taskid -lt $list_size ]
    do
        task_list=$task_list$taskid
        if [ $taskid -ne $((list_size-1)) ]
        then
            task_list=$task_list$delim
        fi
        taskid=$((taskid+1))
    done
    echo $task_list
}

create_delim_separated_optional_reader_task_list()
{
    local list_size=$1
    local delim=$2
    local taskid=0
    local task_list=
    while [ $taskid -lt $list_size ]
    do
        task_list=$task_list${taskid}"-1"
        if [ $taskid -ne $((list_size-1)) ]
        then
            task_list=$task_list$delim
        fi
        taskid=$((taskid+1))
    done
    echo $task_list
}

task_list_for_ml_selection_patch_file()
{
    local num_of_files=$1
    local files_per_task=$2
    local fileno=1
    local taskid=0
    local task_list=
    while [ $fileno -le $num_of_files ]
    do
        task_list=$task_list$taskid
        if [ $fileno -ne $num_of_files ]
        then
            task_list=$task_list:
        fi
        if [ $((fileno % files_per_task)) -eq 0 ]
        then
            taskid=$((taskid+1))
        fi
        fileno=$((fileno+1))
    done
    echo $task_list
}

get_n_random_file_list()
{
    local original_file_list=$1
    local delim=$2
    local num_of_outputs=$3
    file_array=(${original_file_list//$delim/ })
    file_array=($(shuf -e "${file_array[@]}"))
    local fileno=0
    out_file_list=
    while [ $fileno -lt $num_of_outputs ]
    do
        file_name=${file_array[$fileno]}
        out_file_list=$out_file_list$file_name
        if [ $fileno -ne $num_of_outputs ]
        then
            out_file_list=$out_file_list$delim
        fi
        fileno=$((fileno+1))
    done
    echo $out_file_list
}

create_rank_file()
{
    local num_nodes=$1
    local tasks_per_node=$2
    local num_sockets=$3
    local outfilename=$4
    jsrun -n $num_nodes -r 1 -a 1 -c 1 -d packed hostname > hostfile
    readarray -t host_list < hostfile
    hostid=0
    while [ $hostid -lt ${#host_list[@]} ]
    do
        echo ${host_list[hostid]}
        rank=0
        slotid=0
        while [ $rank -lt $tasks_per_node ]
        do
            if [ $rank -eq $(((slotid+1)*tasks_per_node/num_sockets)) ]
            then
                slotid=$((slotid+1))
            fi
            echo "rank "$((rank+tasks_per_node*hostid))"="${host_list[hostid]}" slot="$slotid":"$((rank%(tasks_per_node/num_sockets))) >> $outfilename
            rank=$((rank+1))
        done
        hostid=$((hostid+1))
    done
    rm hostfile
}

min_nodes=$min_num_nodes
max_nodes=$total_num_nodes

# Change the following paths for different environment
gpfs_dir=/p/gpfs1/iopp/temp/wemul_mummi_emulation.$LSB_JOBID
tmpfs_dir=/dev/shm/wemul_mummi_emulation
bb_dir=$BBPATH/wemul_mummi_emulation
mkdir -p $gpfs_dir
jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed mkdir -p $tmpfs_dir
jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed mkdir -p $bb_dir

wemul_home=/usr/WS2/iopp/software/Wemul/codes
wemul_executable=$wemul_home/build/wemul
export LD_LIBRARY_PATH=$wemul_home/libs/AXL/install/lib64:$wemul_home/libs/kvtree/install/lib64:$LD_LIBRARY_PATH

mode=mummi_default

recorder_lib_path=/usr/WS2/iopp/software/Recorder/install/lib/librecorder.so
#recorder_lib_path=/g/g91/chowdhur/codes/Recorder_lassen/Recorder/build/bin/librecorder.so
tracer_lib_path=$recorder_lib_path

num_nodes=$min_nodes

min_iteration=1
max_iteration=10

block_size=1048576
macro_scale_snapshot_filesize=377487360 # 360M each; Each snapshot has three patches
micro_scale_snapshot_filesize=377487360
patch_filesize=125829120 # 120M each
feedback_filesize=62914560 # 60M each

while [ $num_nodes -le $max_nodes ]
do
    printf "\nNum Clients: "
    echo $num_nodes
    printf "\n"

    num_sockets=2
    num_resource_sets=$((num_nodes*num_sockets))
    tasks_per_resource_set=$((tasks_per_node/num_sockets))
    number_of_ranks=$((num_nodes*tasks_per_node))

    # creating rankfiles
    micro_scale_rankfilename=micro_scale_rankfile.$LSB_JOBID.$num_nodes
    rm -f $micro_scale_rankfilename
    create_rank_file $num_nodes $tasks_per_node $num_sockets $micro_scale_rankfilename

    iteration=$min_iteration
    while [ $iteration -le $max_iteration ]
    do
        printf "\nIteration: "
        echo $iteration
        printf "\n"

        # recorder_traces_dir=/p/gpfs1/haridev/iopp/recorder_logs/wemul_mummi_emulation_${mode}_${LSB_JOBID}_${num_nodes}_${iteration}
        recorder_traces_dir=/p/gpfs1/iopp/recorder_logs/mummi_wemul_ppn_${tasks_per_node}.${LSB_JOBID}
        export RECORDER_TRACES_DIR=$recorder_traces_dir

        # Construct list of feedback files' names and related information
        micro_scale_feedback_read_dir=$(generate_uniform_elem_list $gpfs_dir $number_of_ranks)
        micro_scale_feedback_file_list=$(generate_file_list micro_scale_feedback.${num_nodes}.$((iteration-1)). 0 $number_of_ranks)
        micro_scale_feedback_blocksize=$(generate_uniform_elem_list $block_size $number_of_ranks)
        micro_scale_feedback_segcount=$(generate_uniform_elem_list $((feedback_filesize/block_size)) $number_of_ranks)

        # Macro-scale analysis
        macro_scale_snapshot_dir=$gpfs_dir
        macro_scale_snapshot_filename=macro_scale_snapshot.$num_nodes.$iteration
        macro_snapshot_file_segment_count=$((macro_scale_snapshot_filesize/block_size))
        macro_feedback_optional_reader_tasks_list=$(generate_uniform_elem_list $(create_delim_separated_optional_reader_task_list $number_of_ranks ,) $number_of_ranks)
#echo $micro_scale_feedback_read_dir
#echo $micro_scale_feedback_file_list
#echo $macro_feedback_optional_reader_tasks_list
        jsrun -E LD_PRELOAD=$tracer_lib_path -n $num_resource_sets -r $num_sockets -a $tasks_per_resource_set -c $tasks_per_resource_set -d packed $wemul_executable --type data --subtype app --read_input_dirs $micro_scale_feedback_read_dir --read_filenames $micro_scale_feedback_file_list --read_block_size $micro_scale_feedback_blocksize --read_segment_count $micro_scale_feedback_segcount --ranks_per_file_read $macro_feedback_optional_reader_tasks_list --write_input_dirs $macro_scale_snapshot_dir --write_filenames $macro_scale_snapshot_filename --write_block_size $block_size --write_segment_count $macro_snapshot_file_segment_count --shared_file_write

        # Machine learning based patch selection model
        ml_selection_patch_dir=$(generate_uniform_elem_list $gpfs_dir $((number_of_ranks*3)))
        ml_selection_patch_filenames=$(generate_file_list ml_selection_patch_file 0 $((number_of_ranks*3)))
        ml_selection_patch_file_blocksize=$(generate_uniform_elem_list $block_size $((number_of_ranks*3)))
        ml_selection_patch_file_segcount=$(generate_uniform_elem_list $((patch_filesize/block_size)) $((number_of_ranks*3)))
        ml_selection_write_tasks_list=$(task_list_for_ml_selection_patch_file $((number_of_ranks*3)) 3)

        jsrun -E LD_PRELOAD=$tracer_lib_path -n $num_resource_sets -r $num_sockets -a $tasks_per_resource_set -c $tasks_per_resource_set -d packed $wemul_executable --type data --subtype app --read_input_dirs $macro_scale_snapshot_dir --read_filenames $macro_scale_snapshot_filename --read_block_size $block_size --read_segment_count $macro_snapshot_file_segment_count --shared_file_read --write_input_dirs $ml_selection_patch_dir --write_filenames $ml_selection_patch_filenames --write_block_size $ml_selection_patch_file_blocksize --write_segment_count $ml_selection_patch_file_segcount --ranks_per_file_write $ml_selection_write_tasks_list

        # Micro scale simulation
        micro_scale_sim_read_dir=$(generate_uniform_elem_list $gpfs_dir $number_of_ranks)
        micro_scale_sim_filenames=$(get_n_random_file_list $ml_selection_patch_filenames : $number_of_ranks)
        micro_scale_sim_file_blocksize=$(generate_uniform_elem_list $block_size $number_of_ranks)
        micro_scale_sim_file_segcount=$(generate_uniform_elem_list $((patch_filesize/block_size)) $number_of_ranks)
        micro_scale_sim_read_task_list=$(create_delim_separated_task_list $number_of_ranks :)
        micro_scale_sim_write_dir=$tmpfs_dir # write into tmpfs instead of gpfs
        micro_scale_sim_out_filename=micro_scale_sim_out.$num_nodes.$iteration
        micro_scale_sim_out_segcount=$((patch_filesize/block_size))

        mpirun -x "LD_PRELOAD=$tracer_lib_path" -x "RECORDER_TRACES_DIR=$recorder_traces_dir" -rf $micro_scale_rankfilename $wemul_executable --type data --subtype app --read_input_dirs $micro_scale_sim_read_dir --read_filenames $micro_scale_sim_filenames --read_block_size $micro_scale_sim_file_blocksize --read_segment_count $micro_scale_sim_file_segcount --ranks_per_file_read $micro_scale_sim_read_task_list --write_input_dirs $micro_scale_sim_write_dir --write_filenames $micro_scale_sim_out_filename --write_block_size $block_size --write_segment_count $micro_scale_sim_out_segcount --file_per_process_write

        # Micro scale analysis
        micro_scale_an_read_dir=$micro_scale_sim_write_dir
        micro_scale_an_read_filename=$micro_scale_sim_out_filename
        micro_scale_an_read_segcount=$micro_scale_sim_out_segcount
        micro_scale_an_write_dir=$(generate_uniform_elem_list $gpfs_dir $((number_of_ranks+1)))
        micro_scale_snapshot_filename=micro_scale_snapshot.$num_nodes.$iteration
        micro_scale_snapshot_segcount=$((micro_scale_snapshot_filesize/block_size))
        micro_scale_feedback_file_list=$(generate_file_list micro_scale_feedback.${num_nodes}.$iteration. 0 $number_of_ranks)
        micro_scale_an_write_file_list=${micro_scale_snapshot_filename}:${micro_scale_feedback_file_list}
        micro_scale_an_write_blocksize=$(generate_uniform_elem_list $block_size $((number_of_ranks+1)))
        micro_scale_an_write_segcount=${micro_scale_snapshot_segcount}:${micro_scale_feedback_segcount}
        micro_scale_an_write_task_list=$(create_delim_separated_task_list $number_of_ranks ,):$(create_delim_separated_task_list $number_of_ranks :)

        mpirun -x "LD_PRELOAD=$tracer_lib_path" -x "RECORDER_TRACES_DIR=$recorder_traces_dir" -rf $micro_scale_rankfilename $wemul_executable --type data --subtype app --read_input_dirs $micro_scale_an_read_dir --read_filenames $micro_scale_an_read_filename --read_block_size $block_size --read_segment_count $micro_scale_an_read_segcount --file_per_process_read --write_input_dirs $micro_scale_an_write_dir --write_filenames $micro_scale_an_write_file_list --write_block_size $micro_scale_an_write_blocksize --write_segment_count $micro_scale_an_write_segcount --ranks_per_file_write $micro_scale_an_write_task_list

        iteration=$((iteration+1))
        # rm -rf $gpfs_dir/!(*feedback*)
        mkdir -p $gpfs_dir/../feedback_files
        mv $gpfs_dir/*feedback* $gpfs_dir/../feedback_files
        rm -rf $gpfs_dir/*
        mv $gpfs_dir/../feedback_files/* $gpfs_dir
        rm -rf $gpfs_dir/../feedback_files
        jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed rm -rf $tmpfs_dir/*
        jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed rm -rf $bb_dir/*
    done
    rm $micro_scale_rankfilename
    num_nodes=$((num_nodes*2))
done

rm -rf $gpfs_dir
jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed rm -rf $tmpfs_dir
jsrun -n $total_num_nodes -r 1 -a 1 -c 1 -d packed rm -rf $bb_dir
