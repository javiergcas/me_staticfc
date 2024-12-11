# 10/22/2024 - Javier Gonzalez-Castillo
#
# This script will denoise each echo independently
#
#

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
3dcalc -overwrite -a tedana_r01/adaptive_mask.nii.gz -expr 'step(a)' -prefix mask_tedana_at_least_one_echo.nii.gz
3drefit -space MNI mask_tedana_at_least_one_echo.nii.gz

# Scale each echo separately
# --------------------------
echo "++ Scaling volreg versions of each echo:"
echo "========================================"
3dcalc -overwrite -a pb03.${SBJ}.r01.e01.volreg+tlrc -b tedana_r01/dn_ts_e1.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e01.meica_dn
3dcalc -overwrite -a pb03.${SBJ}.r01.e02.volreg+tlrc -b tedana_r01/dn_ts_e2.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e02.meica_dn
3dcalc -overwrite -a pb03.${SBJ}.r01.e03.volreg+tlrc -b tedana_r01/dn_ts_e3.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e03.meica_dn

for EC in e01 e02 e03
do
    echo " + scaling [${EC}]"
    3dTstat -overwrite -prefix rm.mean_pb06.${SBJ}.r01.${EC}.meica_dn pb06.${SBJ}.r01.${EC}.meica_dn+tlrc

    3dcalc -overwrite                                      \
           -a pb06.${SBJ}.r01.${EC}.meica_dn+tlrc          \
           -b rm.mean_pb06.${SBJ}.r01.${EC}.meica_dn+tlrc  \
           -c mask_epi_extents+tlrc                        \
           -expr 'c * min(200, a/b*100)*step(a)*step(b)'   \
           -prefix pb07.${SBJ}.r01.${EC}.meica_dn.scale
    rm rm.mean_pb06.${SBJ}.r01.${EC}.meica_dn+tlrc.*
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
               -input pb07.${SBJ}.r01.${EC}.meica_dn.scale+tlrc                       \
               -censor censor_${SBJ}_combined_2.1D                                    \
               -cenmode ${INTERP_MODE}                                                \
               -ort X.nocensor.xmat.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_${INTERP_MODE}  \
               -mask mask_tedana_at_least_one_echo.nii.gz
  done
done

for EC in e01 e02 e03
do
   echo " + Denoising echo [${EC} | ALL]"
   3dTproject -overwrite                                                              \
               -polort 0                                                              \
               -input pb07.${SBJ}.r01.${EC}.meica_dn.scale+tlrc                       \
               -ort X.nocensor.xmat.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_ALL             \
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
              -inset  errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_${INTERP_MODE}+tlrc          \
              -prefix errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_${INTERP_MODE}.${ATLAS_NAME}

    3dROIstats -quiet \
               -mask ${ATLAS_PATH} \
               errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_${INTERP_MODE}+tlrc > errts.${SBJ}.r01.${EC}.meica_dn.scale.tproject_${INTERP_MODE}.${ATLAS_NAME}_000.netts
  done
done

echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="