#!/bin/bash
#
# NOTE:
# This version of the NORDIC code does not seem to understand the DIROUT argument, meaning if I try to pass it
# then not all files get written. The workaround it is to not use DIROUT, but to cd into the target directory first.

INPUT_PATH=$1
OUTPUT_PATH=$2

OUTPUT_FOLDER=`dirname ${OUTPUT_PATH}`
OUTPUT_FILE=`basename ${OUTPUT_PATH}`

echo "++ Input Path: ${INPUT_PATH}"
echo "++ Output Path: ${OUTPUT_PATH}"

module load matlab

cd ${OUTPUT_FOLDER}
pwd
matlab -nojvm<<-EOF

   close all
   clear all
   
   addpath('/data/SFIMJGC_Introspec/NORDIC_Raw/')
   
   ARG.temporal_phase=1;
   ARG.phase_filter_width=3;
   ARG.gfactor_patch_overlap=6;
   ARG.magnitude_only = 1; 
   ARG.noise_volume_last = 0;
   ARG.save_residual_matlab=1;
   ARG.phase_filter_width=10;
   %ARG.DIROUT='$OUTPUT_FOLDER';
   ARG.write_gzipped_niftis=1;
   ARG.save_add_info=1;
   ARG.save_gfactor_map=1; 
   % Potentially useful options
   %ARG.kernel_size_PCA = [10 10 10];
   
   %%
   file_magn  = '$INPUT_PATH'
   file_phase = './DOESNOTEXISTS.nii.gz'
   file_out   = '$OUTPUT_FILE'
   NIFTI_NORDIC(file_magn,file_phase,file_out,ARG)

EOF
