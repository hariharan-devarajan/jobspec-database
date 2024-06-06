#!/bin/bash
#PBS -l nodes=1:ppn=8,walltime=24:00:00
#PBS -N app-bids-mrtrix3_connectome

function json_arr_to_bash_list {
    local v=$@
    if [[ $v =~ \[.*\] ]]; then
        v=$(echo $v | tr -d '\[\],')  
    fi
    echo $v
}
for prm in $(jq -r 'keys[]' config.json | grep -Ev ^_); do
    val=$(json_arr_to_bash_list $(jq -r ".$prm" config.json))
    eval "$prm=( $val )"
done

if [[ $preprocessed == true ]]; then
    pp_arg="--preprocessed"
fi
if [[ -n $streamlines && $streamlines != null ]]; then
    sl_arg="--streamlines $streamlines"
fi

#BIDS
function check_required_fields {
    #these fields are required for preprocessed=true
    local json=$1
    local f1=$(jq '.TotalReadoutTime' $json)
    local f2=$(jq '.PhaseEncodingDirection' $json)
    [[ -n $f1 && -n $f2 && $f1 != null && $f2 != null ]]
}

function copy_to_bids {
    # inputs:   (1) bids subdirectory, (2) bids image suffix,
    #           (3) BL variables name, (4) optinal run # suffix

    local sdir=$1
    local itype=$2
    local vname=$3
    local vind=$4
    local suffix=$5
    local msg meta metaval

    if [[ -z $vind ]]; then
        vind=0
    fi
    if [[ -n $suffix ]]; then
        suffix=${suffix}_
    fi

    local bids_dir=input/${bids_sub}/$sdir
    local bids_name=$suffix$itype
    local vvalue=$(eval echo \${$vname[$vind]})

    mkdir -p $bids_dir
    ln -s $(pwd)/$vvalue ${bids_dir}/${bids_sub}_${bids_name}.nii.gz
    #make BIDS json sidecar from BL metadata
    jq --arg idprm $vname --argjson idx $vind \
        '[ ._inputs[] | select(.id == $idprm) | .meta ] | .[$idx]' \
        config.json > ${bids_dir}/${bids_sub}_${bids_name}.json
    if [[ $sdir == dwi ]]; then
        if [[ $preprocessed == false ]] && \
            ! check_required_fields ${bids_dir}/${bids_sub}_${bids_name}.json
        then
            msg="TotalReadoutTime and PhaseEncodingDirection BIDS fields reqd "
            msg+="to run mrtrix3_connectome preproc. Missing for ${!vname}"
            echo $msg
            exit 1
        fi
        for meta in bvec bval; do
            #BL names with "s"
            metaval=$(eval echo \${${vname/dwi/${meta}s}[$vind]}) 
            ln -s $(pwd)/$metaval ${bids_dir}/${bids_sub}_${bids_name}.$meta
        done
    fi
}

sub=$(jq -r "._inputs[0].meta.subject" config.json | tr -d "_")
if [[ -z $sub || $sub == null ]]; then
    sub=0
fi
bids_sub=sub-$sub

#clean up previous job (just in case)
rm -rf ${bids_sub} output input

copy_to_bids anat T1w t1

if [[ ${#dwi[@]} -gt 1 ]]; then
    for ((i=0; i<${#dwi[@]}; i++)); do
        run_suffix=run-$(printf '%02d' $((i+1))) 
        copy_to_bids dwi dwi dwi $i $run_suffix
    done
elif [[ $preprocessed == false ]]; then
    echo "Two DWI images are required to run mrtrix3_connectome preprocessing."
    exit 1
else
    copy_to_bids dwi dwi dwi 0
fi

singularity run -e \
    docker://katealpert/bids-mrtrix3_connectome \
    input . participant \
    -n 8 --parcellation $parcellation $sl_arg $pp_arg \
    --participant_label $sub
res=$?

if [[ -d ${bids_sub} && $res == 0 ]]; then
    mv ${bids_sub} output
    rm -rf input
fi

#exit code from the last command (singularity) will be used.
exit $res
