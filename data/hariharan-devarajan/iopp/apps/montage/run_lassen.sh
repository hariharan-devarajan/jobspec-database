#!/bin/bash
#BSUB -cwd /usr/workspace/iopp/software/iopp/apps/montage
#BSUB -o out.log
#BSUB -e err.log

#BSUB -G asccasc
#BSUB -q pbatch
#BSUB -nnodes 32
#BSUB -W 06:00 
#BSUB -J montage_100T
num_nodes=32
tasks_per_node=40

echo "My job id is $LSB_JOBID"
date=$(date '+%Y%m%d')
times=100
montage_bin_dir=/usr/workspace/iopp/applications/Montage/install/bin
montage_data_dir=/p/gpfs1/haridev/iopp/datasets/montage/m101/${times}T
montage_run_dir=/p/gpfs1/haridev/iopp/datasets/montage/m101/${times}T_mosaic
mkdir -p $montage_run_dir
processes=$((num_nodes * tasks_per_node))
export RECORDER_TRACES_DIR=/p/gpfs1/haridev/iopp/recorder_logs/recorder-$date/montage_haridev_$LSB_JOBID

unset RECORDER_NO_MPI
recorder_lib_path=/usr/workspace/iopp/software/Recorder/install/lib/librecorder.so
tracer_lib_path=$recorder_lib_path

pushd $montage_run_dir

echo "Create directories to hold processed images"
mkdir Kprojdir diffdir corrdir

export LD_PRELOAD=$tracer_lib_path

echo "Create a metadata table of the input images, Kimages.tbl"
RECORDER_NO_MPI=1 $montage_bin_dir/mImgtbl $montage_data_dir Kimages.tbl

echo "Create a FITS header describing the footprint of the mosaic"
RECORDER_NO_MPI=1 $montage_bin_dir/mMakeHdr Kimages.tbl Ktemplate.hdr

echo "Reproject the input images"
# lrun -N$num_nodes -T$tasks_per_node --env "LD_PRELOAD=$tracer_lib_path" $montage_bin_dir/mProjExecMPI -p $montage_data_dir Kimages.tbl Ktemplate.hdr Kprojdir Kstats.tbl
RECORDER_NO_MPI=1 $montage_bin_dir/mProjExec -p $montage_data_dir Kimages.tbl Ktemplate.hdr Kprojdir Kstats.tbl

echo "Create a metadata table of the reprojected images"
RECORDER_NO_MPI=1 $montage_bin_dir/mImgtbl Kprojdir/ images.tbl

echo "Coadd the images to create a mosaic without background corrections"
jsrun -p $processes --rs_per_host=$tasks_per_node $montage_bin_dir/mAddMPI -p Kprojdir/ images.tbl Ktemplate.hdr m17_uncorrected.fits

echo "Make a PNG of the mosaic for visualization"
RECORDER_NO_MPI=1 $montage_bin_dir/mViewer -ct 1 -gray m17_uncorrected.fits -1s max gaussian-log -out m17_uncorrected.png

echo "Analyze the overlaps between images"

echo "Find out which images overlap"
RECORDER_NO_MPI=1 $montage_bin_dir/mOverlaps images.tbl diffs.tbl

echo "Subtract each pair of overlapping images and create a set of difference images in the diffdr subdirectory"
RECORDER_NO_MPI=1 $montage_bin_dir/mDiffExec -p Kprojdir/ diffs.tbl Ktemplate.hdr diffdir

echo "Calculate plane-fitting coefficients for each difference image"
RECORDER_NO_MPI=1 $montage_bin_dir/mFitExec diffs.tbl fits.tbl diffdir

echo "Perform background modeling and compute corrections for each image"
RECORDER_NO_MPI=1 $montage_bin_dir/mBgModel images.tbl fits.tbl corrections.tbl

echo "Apply corrections to each image"
RECORDER_NO_MPI=1 $montage_bin_dir/mBgExec -p Kprojdir/ images.tbl corrections.tbl corrdir


echo "Coadd the images to create a mosaic with background corrections"
jsrun -p $processes --rs_per_host=$tasks_per_node --env "LD_PRELOAD=$tracer_lib_path" $montage_bin_dir/mAddMPI -p corrdir/ images.tbl Ktemplate.hdr m17.fits


echo "Make a PNG of the corrected mosaic for visualization"
RECORDER_NO_MPI=1 $montage_bin_dir/mViewer -ct 1 -gray m17.fits -1s max gaussian-log -out m17.png

echo "Move all the output to a directory"
export timestamp=$(date +%s)
export out_dir=out_${timestamp}
mkdir ../$out_dir
mv * ../$out_dir

popd
