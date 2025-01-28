# 10/22/2024 - Javier Gonzalez-Castillo
#
# This script will do teh regression of common nuisances in the volreg and extract
# FC and timeseries

export OMP_NUM_THREADS=32
export AFNI_COMPRESSOR=GZIP

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

echo "++ Scaling data post volreg"
echo "++ ------------------------"
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
echo "++ -------------------------------------------------------------------------------"
for EC in e01 e02 e03
do
    # Global regression with no detrending prior to computing GS
    3dROIstats -quiet \
               -mask mask_tedana_at_least_one_echo.nii.gz \
                pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc.HEAD | awk '{print $1}' > pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D
    
#    # Alternative GS regression computing GS post detrending           
#    3dDetrend -overwrite \
#              -polort 5 \
#              -prefix pb03.${SBJ}.r01.${EC}.volreg.scale.dt5 \
#                      pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc
#    3dROIstats -quiet \
#               -mask mask_tedana_at_least_one_echo.nii.gz \
#               pb03.${SBJ}.r01.${EC}.volreg.scale.dt5+tlrc.HEAD | awk '{print $1}' > pb03.${SBJ}.r01.${EC}.volreg.scale.GSdt5.1D
#    rm pb03.${SBJ}.r01.${EC}.volreg.scale.dt5+tlrc.*
done

echo "++ Denoising each echo separately (Basic & GS Pipelines)"
echo "++ ------------------------------------------------------"

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
               -ort X.nocensor.xmat.1D                                                      \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic    \
               -mask mask_tedana_at_least_one_echo.nii.gz
    echo " + Denoising echo [${EC} | ${INTERP_MODE}] | GSasis"
    3dTproject -overwrite                                                                 \
               -polort 0                                                                  \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                             \
               -censor censor_${SBJ}_combined_2.1D                                        \
               -cenmode ${INTERP_MODE}                                                    \
               -ort X.nocensor.xmat.1D                                                    \
               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D                          \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis \
               -mask mask_tedana_at_least_one_echo.nii.gz
    echo " + Denoising echo [${EC} | ${INTERP_MODE}] | GSdt5"
#    3dTproject -overwrite                                                                 \
#               -polort 0                                                                  \
#               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                             \
#               -censor censor_${SBJ}_combined_2.1D                                        \
#               -cenmode ${INTERP_MODE}                                                    \
#               -ort X.nocensor.xmat.1D                                                    \
#               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSdt5.1D                           \
#               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSdt5  \
#               -mask mask_tedana_at_least_one_echo.nii.gz
  done
done

# Next we do this keeping all timepoins (i.e., no censoring)
for EC in e01 e02 e03
do
   echo " + Denoising echo [${EC} | ALL]"
   3dTproject -overwrite                                                              \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -ort X.nocensor.xmat.1D                                                \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL_Basic         \
               -mask mask_tedana_at_least_one_echo.nii.gz
    echo " + Denoising echo [${EC} | ALL] | GSasis"
    3dTproject -overwrite                                                             \
               -polort 0                                                              \
               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
               -ort X.nocensor.xmat.1D                                                \
               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSasis.1D                      \
               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL_GSasis        \
               -mask mask_tedana_at_least_one_echo.nii.gz
#    echo " + Denoising echo [${EC} | ALL] | GSdt5"
#    3dTproject -overwrite                                                             \
#               -polort 0                                                              \
#               -input pb03.${SBJ}.r01.${EC}.volreg.scale+tlrc                         \
#               -ort X.nocensor.xmat.1D                                                \
#               -ort pb03.${SBJ}.r01.${EC}.volreg.scale.GSdt5.1D                       \
#               -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_ALL_GSdt5         \
#               -mask mask_tedana_at_least_one_echo.nii.gz
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
    3dNetCorr -overwrite                                                                          \
              -mask mask_tedana_at_least_one_echo.nii.gz                                          \
              -in_rois ${ATLAS_PATH}                                                              \
              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic+tlrc            \
              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}

    3dROIstats -quiet \
               -mask ${ATLAS_PATH} \
               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic+tlrc > errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_Basic.${ATLAS_NAME}_000.netts

    echo " + Extracting ROI Timeseries and connectivity for [${EC} + Basic Denoising + GSR (asis) ]"
    3dNetCorr -overwrite                                                                               \
              -mask mask_tedana_at_least_one_echo.nii.gz                                               \
              -in_rois ${ATLAS_PATH}                                                                   \
              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis+tlrc          \
              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis.${ATLAS_NAME}

    3dROIstats -quiet \
               -mask ${ATLAS_PATH} \
               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis+tlrc > errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSasis.${ATLAS_NAME}_000.netts

#    echo " + Extracting ROI Timeseries and connectivity for [${EC} + Basic Denoising + GSR (dt5) ]"
#    3dNetCorr -overwrite                                                                               \
#              -mask mask_tedana_at_least_one_echo.nii.gz                                               \
#              -in_rois ${ATLAS_PATH}                                                                   \
#              -inset  errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSdt5+tlrc          \
#              -prefix errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSdt5.${ATLAS_NAME}
#
#    3dROIstats -quiet \
#               -mask ${ATLAS_PATH} \
#               errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSdt5+tlrc > #errts.${SBJ}.r01.${EC}.volreg.scale.tproject_${INTERP_MODE}_GSdt5.${ATLAS_NAME}_000.netts

  done
done

# Remove all the versions of GS
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ALL_GSasis+tlrc.*
#rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ALL_GSdt5+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_NTRP_GSasis+tlrc.*
#rm errts.${SBJ}.r01.e??.volreg.scale.tproject_NTRP_GSdt5+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ZERO_GSasis+tlrc.*
#rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ZERO_GSdt5+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_KILL_GSasis+tlrc.*
#rm errts.${SBJ}.r01.e??.volreg.scale.tproject_KILL_GSdt5+tlrc.*

# Remove the denoised data without GS regression
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_ZERO_Basic+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_KILL_Basic+tlrc.*
rm errts.${SBJ}.r01.e??.volreg.scale.tproject_NTRP_Basic+tlrc.*
pwd
echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="