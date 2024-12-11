# 10/22/2024 - Javier Gonzalez-Castillo
#
# This script will do teh regression of common nuisances in the volreg and extract
# FC and timeseries

export OMP_NUM_THREADS=32
set -e

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024/'
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data`

SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`
FMRI_DATA_DIR=`echo ${SBJ_DIR}/D02_Preproc_fMRI_${SES}`
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

echo "++ Scaling data..."
echo "=================="
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

# Project MEICA components from each echo separately
# --------------------------------------------------
echo "++ Denoising each echo separately (using MEICA bad components)"
echo "=============================================================="
for EC in e01 e02 e03
do
  for INTERP_MODE in ZERO KILL NTRP
  do
    echo " + Denoising echo [${EC} | ${INTERP_MODE}]"
    3dTproject -overwrite                                                             \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -censor censor_${SBJ}_combined_2.1D                                    \
               -cenmode ${INTERP_MODE}                                                \
               -ort X.nocensor.xmat.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}    \
               -mask mask_tedana_at_least_one_echo.nii.gz
  done
done

for EC in e01 e02 e03
do
   echo " + Denoising echo [${EC} | ALL]"
   3dTproject -overwrite                                                              \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -ort X.nocensor.xmat.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL               \
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
    echo " + Extracting ROI Timeseries and connectivity for [${EC}]"
    3dNetCorr -overwrite                                                                          \
              -mask mask_tedana_at_least_one_echo.nii.gz                                          \
              -in_rois ${ATLAS_PATH}                                                              \
              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}+tlrc            \
              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}.${ATLAS_NAME}

    3dROIstats -quiet \
               -mask ${ATLAS_PATH} \
               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}+tlrc > errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}.${ATLAS_NAME}_000.netts
  done
done

echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="