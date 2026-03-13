#!/bin/bash

set -e 
if [ -z "$SBJ" ]; then
   echo "SBJ is missing. Exit now"
   exit
fi
if [ -z "$SES" ]; then
   echo "SES is missing. Exit now"
   exit
fi

PRJ_DIR=/data/SFIMJGC_HCP7T/BCBL2024/
PRCS_DATA_DIR=`echo ${PRJ_DIR}/prcs_data`
FIGURES_DIR=`echo ${PRJ_DIR}/me_staticfc/code/notebooks/figures/`
echo "++ INFO: Subject = ${SBJ}"
echo "++ INFO: Session = ${SES}"
echo "++ INFO: PRJ_DIR = ${PRJ_DIR}"
echo "++ INFO: PRCS_DATA_DIR = ${PRCS_DATA_DIR}"
echo "++ INFO: FIGURES_DIR = ${FIGURES_DIR}"


cd ${PRCS_DATA_DIR}/${SBJ}/
if [ ! -d NORDIC_Diff_Analysis_SuppFig04-${SES} ]; then 
   echo "++ INFO: Creating new folder NORDIC_Diff_Analysis_SuppFig04-${SES}"
   mkdir NORDIC_Diff_Analysis_SuppFig04-${SES}; 
fi

echo "++ Entering working folder"
cd NORDIC_Diff_Analysis_SuppFig04-${SES}

# Create Differential errts NORDIC on/off following basic pre-processing
# ======================================================================
3dcalc -overwrite -a ../D03_Preproc_${SES}_NORDIC-off/errts.${SBJ}.r01.e02.volreg.tproject_ALL_Basic+tlrc. \
                  -b ../D03_Preproc_${SES}_NORDIC-on/errts.${SBJ}.r01.e02.volreg.tproject_ALL_Basic+tlrc.  \
                  -overwrite -expr 'a-b' -prefix diff.errts_ALL_Basic.e02.${SBJ}.${SES}.nii

# Create Differential errts NORDIC on/off right after alignment operations (no regression)
# ========================================================================================
3dcalc -overwrite -a ../D03_Preproc_${SES}_NORDIC-off/pb03.${SBJ}.r01.e02.volreg+tlrc. \
                  -b ../D03_Preproc_${SES}_NORDIC-on/pb03.${SBJ}.r01.e02.volreg+tlrc.  \
                  -overwrite -expr 'a-b' -prefix diff.pb03.e02.${SBJ}.${SES}.nii

# Create Mean and StDev Maps for Supp Fig 04 (following no regression)
# ====================================================================
3dTstat -overwrite -mean -stdev -cvar -cvarinv -prefix diff.pb03.e02.${SBJ}.${SES}.Tstats.nii diff.pb03.e02.${SBJ}.${SES}.nii

# Seed-based Correlation from Visual Cortex 
# =========================================

# Create seed
printf "%s\n" "7.5 -85.5 7.5" > seed_xyz.1D
3dUndump -overwrite -master diff.pb03.e02.${SBJ}.${SES}.nii \
         -srad 3 -xyz seed_xyz.1D 

3dcopy -overwrite undump+tlrc seed_sphere_${SBJ}.nii
rm undump+tlrc.*

# Corr Analysis in Differential Map following  No-Regression Data
3dmaskave -quiet -mask seed_sphere_${SBJ}.nii \
          diff.pb03.e02.${SBJ}.${SES}.nii > pb03.seed_ts_${SBJ}.${SES}.1D
3dTcorr1D -overwrite -pearson -Fisher -prefix pb03.seedcorr_${SBJ}.${SES}.nii \
          diff.pb03.e02.${SBJ}.${SES}.nii pb03.seed_ts_${SBJ}.${SES}.1D

# Corr Analysis in Differential Map following Basic Regression 
3dmaskave -quiet -mask seed_sphere_${SBJ}.nii \
          diff.errts_ALL_Basic.e02.${SBJ}.${SES}.nii > ALL_Basic.seed_ts_${SBJ}.${SES}.1D
3dTcorr1D -overwrite -pearson -Fisher -prefix ALL_Basic.seedcorr_${SBJ}.${SES}.nii \
          diff.errts_ALL_Basic.e02.${SBJ}.${SES}.nii ALL_Basic.seed_ts_${SBJ}.${SES}.1D

# Copy Anatomical Data for Underlay Visualization 
# ===============================================
3dcopy -overwrite ../D03_Preproc_${SES}_NORDIC-off/anat_final.${SBJ}+tlrc anat_final.${SBJ}.${SES}.nii

# Take Screenshots
# ================

# Mean Image across time for Diff Maps
@chauffeur_afni \
  -ulay diff.pb03.e02.${SBJ}.${SES}.Tstats.nii'[0]' \
  -prefix ${FIGURES_DIR}/pBOLD_SuppFig04_${SBJ}_${SES}_mean \
  -montx 1 -monty 1 \
  -ulay_range -200 200 \
  -set_dicom_xyz -34.5 76.5 10.5 \
  -set_xhairs OFF \
  -label_mode 1 -label_size 4 \
  -blowup 6 \
  -do_clean

# StDev Image across time for Diff Maps
@chauffeur_afni \
  -ulay diff.pb03.e02.${SBJ}.${SES}.Tstats.nii'[1]' \
  -prefix ${FIGURES_DIR}/pBOLD_SuppFig04_${SBJ}_${SES}_stdv \
  -montx 1 -monty 1 \
  -ulay_range -200 200 \
  -set_dicom_xyz -34.5 76.5 10.5 \
  -set_xhairs OFF \
  -label_mode 1 -label_size 4 \
  -blowup 6 \
  -do_clean
echo "======================================================"
pwd
# Correlation Map (No Regression)
@chauffeur_afni \
  -ulay anat_final.${SBJ}.${SES}.nii \
  -olay pb03.seedcorr_${SBJ}.${SES}.nii \
  -prefix ../../../me_staticfc/code/notebooks/figures/pBOLD_SuppFig04_${SBJ}_${SES}_no_reg_sbco \
  -montx 1 -monty 1 \
  -set_dicom_xyz -7.0 85.0 9.0 \
  -set_xhairs OFF \
  -label_mode 1 -label_size 4 \
  -cbar GoogleTurbo \
  -func_range 0.7 \
  -thr_olay 0.4 \
  -olay_alpha Yes \
  -olay_boxed Yes \
  -blowup 6 \
  -do_clean
# Correlation Map (After Basic Denoising)
@chauffeur_afni \
  -ulay anat_final.${SBJ}.${SES}.nii \
  -olay ALL_Basic.seedcorr_${SBJ}.${SES}.nii \
  -prefix ../../../me_staticfc/code/notebooks/figures/pBOLD_SuppFig04_${SBJ}_${SES}_basic_reg_sbco \
  -montx 1 -monty 1 \
  -set_dicom_xyz -7.0 85.0 9.0 \
  -set_xhairs OFF \
  -label_mode 1 -label_size 4 \
  -cbar GoogleTurbo \
  -func_range 0.7 \
  -thr_olay 0.4 \
  -olay_alpha Yes \
  -olay_boxed Yes \
  -blowup 6 \
  -do_clean

echo "++ ============================================================ ++"
echo "++ ======  Script Finished Correctly ========================== ++"
echo "++ ============================================================ ++"
