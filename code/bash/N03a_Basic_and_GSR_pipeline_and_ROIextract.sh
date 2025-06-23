#!/bin/bash
# 10/22/2024 - Javier Gonzalez-Castillo
# 06/06/2025 - JGC Script now computes TSNR on errts files
# This script will do teh regression of common nuisances in the volreg and extract
# FC and timeseries

export OMP_NUM_THREADS=32
export AFNI_COMPRESSOR=GZIP

set -e

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024/'
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data`

SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`

# NOTE: This needs to change once we re-organize folders
# Trick to pick results for NORDIC pipeline without having to re-write other notebooks
if [[ -z ${AFNIPROC_OUTDIR} ]]; then
   echo "++ WARNING: Variable {AFNIPROC_OUTDIR} not provided --> setting FMRI_DATA_DIR to default value"
   FMRI_DATA_DIR=`echo ${SBJ_DIR}/D02_Preproc_fMRI_${SES}`
   echo "            FMRI_DATA_DIR=${FMRI_DATA_DIR}"
else
   FMRI_DATA_DIR=`echo ${AFNIPROC_OUTDIR}/`
   echo "++ INFO: Setting FMRI_DATA_DIR=${FMRI_DATA_DIR} as provided via AFNIPROC_OUTDIR"
   cd ${FMRI_DATA_DIR} 
   ln -fs ${SBJ_DIR}/D02_Preproc_fMRI_${SES}/mask_tedana_at_least_one_echo.nii.gz ${FMRI_DATA_DIR}/mask_tedana_at_least_one_echo.nii.gz
fi
# End of trick

ATLAS_PATH=`echo ${ATLASES_DIR}/${ATLAS_NAME}/${ATLAS_NAME}.nii.gz`

echo "++ Working on Subject ${SBJ}/${SES}... post afni proc"
echo " + PRJDIR=${PRJDIR}"
echo " + PRCS_DATA_DIR=${PRCS_DATA_DIR}"
echo " + SBJ_DIR=${SBJ_DIR}"
echo " + FMRI_DATA_DIR=${FMRI_DATA_DIR}"
echo "=============================================================================="

# Enter destination folder
# ------------------------
echo "++ Entering FMRI_DATA_DIR"
echo "========================="
cd ${FMRI_DATA_DIR}
echo " +  `pwd`"

echo "++ Scaling data post volreg"
echo "++ ========================"
for EC in e01 e02 e03
do
    echo " + scaling [${EC}]"
    3dTstat -overwrite -prefix rm.mean_pb03.${SBJ}.r01.${EC}.volreg pb03.${SBJ}.r01.${EC}.volreg+tlrc

    3dcalc -overwrite                                      \
           -a pb03.${SBJ}.r01.${EC}.volreg+tlrc            \
           -b rm.mean_pb03.${SBJ}.r01.${EC}.volreg+tlrc    \
           -c mask_epi_extents+tlrc                        \
           -expr 'c * min(200, a/b*100)*step(a)*step(b)'   \
           -prefix pb03.${SBJ}.r01.${EC}.volreg.scale
    rm rm.mean_pb03.${SBJ}.r01.${EC}.volreg+tlrc.*
done

echo "++ Compute GS in two different ways for each echo separately post volreg & scaling"
echo "++ ==============================================================================="
for EC in e01 e02 e03
do
    # Global regression with no detrending prior to computing GS
    3dROIstats -quiet \
               -mask mask_tedana_at_least_one_echo.nii.gz \
                pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc.HEAD | awk '{print $1}' > pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D
done

echo "++ Denoising each echo separately (Basic & GS Pipelines)"
echo "++ ====================================================="

# NOTE: Afni_proc computes CompCorre regressors based on OC data, which can be influenced by whether or not NORDIC is in place (e.g., S0 and t2s maps can differ)
# To avoid this, we need to compute our own CompCorr regressors independently of NORDIC --> compute them on each echo --> create a GLM matrix per echo, not use the one for afni_proc.py

echo "++ 1. Computing CompCorr regressors per echo...."
echo "++ ---------------------------------------------"
tr_counts=`3dinfo -nt pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc`
# Create Echo Specific CompCorr regressors
for EC in e01 e02 e03
do
    # to censor, create per-run censor files
    1d_tool.py -overwrite -set_run_lengths ${tr_counts} -select_runs 01 -infile censor_${SBJ}_combined_2.1D -write rm.censor.${EC}.r01.1D
    # do not let censored time points affect detrending
    3dTproject -overwrite -polort 5 -prefix rm.det_pcin_${EC}_r01 -censor rm.censor.${EC}.r01.1D -cenmode KILL -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc
    # make ROI PCs (per run) : FSvent
    3dpc -overwrite -mask follow_ROI_FSvent+tlrc -pcsave 3 -prefix rm.ROIPC.FSvent.${EC}.r01 rm.det_pcin_${EC}_r01+tlrc
    # zero pad censored TRs and further pad to fill across all runs
    1d_tool.py -overwrite -censor_fill_parent rm.censor.${EC}.r01.1D -infile rm.ROIPC.FSvent.${EC}.r01_vec.1D  -write - | 1d_tool.py -overwrite -set_run_lengths ${tr_counts} -pad_into_many_runs 01 1 -infile - -write ROIPC.FSvent.${EC}.r01.1D
done

echo "++ 2. Create GLM matrix per-echo for later use in 3dTproject..."
echo "++ ------------------------------------------------------------"
# Create GLM matrix per-echo
for EC in e01 e02 e03
do
    3dDeconvolve -overwrite                                                   \
        -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc.HEAD                   \
        -censor censor_${SBJ}_combined_2.1D                                   \
        -ortvec bandpass_rall.1D bandpass                                     \
        -ortvec ROIPC.FSvent.${EC}.r01.1D ROIPC.FSvent.r01                    \
        -ortvec mot_demean.r01.1D mot_demean_r01                              \
        -ortvec mot_deriv.r01.1D mot_deriv_r01                                \
        -polort 5                                                             \
        -num_stimts 0                                                         \
        -jobs 32                                                              \
        -fout -tout -x1D X.xmat.${EC}.1D -xjpeg X.${EC}.jpg                   \
        -x1D_uncensored X.nocensor.xmat.${EC}.1D                              \
        -fitts fitts.${EC}.${SBJ}                                             \
        -errts errts.${EC}.${SBJ}                                             \
        -x1D_stop                                                             \
        -cbucket all_betas.${EC}.${SBJ}                                       \
        -bucket stats.${EC}.${SBJ}
done


# First we will do this with the different interpolation schemes
for EC in e01 e02 e03
do
  for INTERP_MODE in ZERO KILL NTRP
  do
    echo " + Denoising echo [${EC} | ${INTERP_MODE}]"
    3dTproject -overwrite                                                                   \
               -polort 0                                                                    \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                               \
               -censor censor_${SBJ}_combined_2.1D                                          \
               -cenmode ${INTERP_MODE}                                                      \
               -ort X.nocensor.xmat.${EC}.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic    \
               -mask mask_tedana_at_least_one_echo.nii.gz
    echo " + Denoising echo [${EC} | ${INTERP_MODE}] | GSasis"
    3dTproject -overwrite                                                                 \
               -polort 0                                                                  \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                             \
               -censor censor_${SBJ}_combined_2.1D                                        \
               -cenmode ${INTERP_MODE}                                                    \
               -ort X.nocensor.xmat.${EC}.1D                                              \
               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D                          \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis \
               -mask mask_tedana_at_least_one_echo.nii.gz
  done
done

# Next we do this keeping all timepoins (i.e., no censoring)
for EC in e01 e02 e03
do
  echo " + Denoising echo [${EC} | ALL]"
   3dTproject -overwrite                                                              \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -ort X.nocensor.xmat.${EC}.1D                                          \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL_Basic         \
               -mask mask_tedana_at_least_one_echo.nii.gz
    echo " + Denoising echo [${EC} | ALL] | GSasis"
    3dTproject -overwrite                                                             \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -ort X.nocensor.xmat.${EC}.1D                                          \
               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D                      \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL_GSasis        \
               -mask mask_tedana_at_least_one_echo.nii.gz
done

# Extract ROI Timeseries
# ----------------------
echo "++ Extracting ROI Timeseries per echo for atlas [${ATLAS_NAME}]"
echo "==============================================================="
for EC in e01 e02 e03
do
  for INTERP_MODE in ZERO KILL NTRP ALL
  do
    echo " + Extracting ROI Timeseries and connectivity for [${EC} + Basic Denoising]"
    echo "3dNetCorr -overwrite -mask mask_tedana_at_least_one_echo.nii.gz -in_rois ${ATLAS_PATH} -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic+tlrc -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}"
    3dNetCorr -overwrite                                                                                \
              -push_thru_many_zeros                                                                     \
              -mask mask_tedana_at_least_one_echo.nii.gz                                                \
              -in_rois ${ATLAS_PATH}                                                                    \
              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic+tlrc            \
              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}

    3dROIstats -quiet \
               -mask ${ATLAS_PATH} \
               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic+tlrc > errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}_000.netts

    echo " + Extracting ROI Timeseries and connectivity for [${EC} + Basic Denoising + GSR (asis) ]"
    3dNetCorr -overwrite                                                                                \
              -push_thru_many_zeros                                                                     \
              -mask mask_tedana_at_least_one_echo.nii.gz                                                \
              -in_rois ${ATLAS_PATH}                                                                    \
              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis+tlrc           \
              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis.${ATLAS_NAME}

    3dROIstats -quiet                                                                                   \
               -mask ${ATLAS_PATH}                                                                      \
               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis+tlrc > errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis.${ATLAS_NAME}_000.netts
  done
done

# Compute TSNR
# ------------
echo "++ Computing Full Brain TSNR for Basic and GSasis"
echo "================================================="
for EC in e01 e02 e03
do
  for SCENARIO in ALL_Basic ALL_GSasis
  do
      3dTstat -overwrite -mean -prefix rm.signal.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc 
      3dTstat -overwrite -stdev -prefix rm.noise.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}+tlrc
      3dcalc -overwrite -a rm.signal.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii -b rm.noise.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii -expr 'a/b' -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.TSNR.nii
      # Compute TSNR at the whole-brain level
      compute_ROI_stats.tcsh                                                           \
         -out_dir    tsnr_stats_regress                                                \
         -stats_file tsnr_stats_regress/TSNR_FB_${EC}_${SCENARIO}.txt                  \
         -dset_ROI   mask_epi_anat.${SBJ}+tlrc                                         \
         -dset_data  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.TSNR.nii \
         -rset_label brain                                                             \
         -rval_list  1

     # Compute TSNR at the ROI level (same ROIs used by afni_proc for the report)
     compute_ROI_stats.tcsh                                                            \
         -out_dir    tsnr_stats_regress                                                \
         -stats_file tsnr_stats_regress/TSNR_ROIs_${EC}_${SCENARIO}.txt                \
         -dset_ROI   ROI_import_MNI_2009c_asym_resam+tlrc                              \
         -dset_data  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.TSNR.nii \
         -rset_label MNI_2009c_asym                                                    \
         -rval_list  ALL_LT

     rm rm.signal.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii rm.noise.errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${SCENARIO}.nii
  done
done

# Remove all the versions of GS
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_NTRP_GSasis+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ZERO_GSasis+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_KILL_GSasis+tlrc.*

# Remove the denoised data without GS regression
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ZERO_Basic+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_KILL_Basic+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_NTRP_Basic+tlrc.*
pwd
echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="
