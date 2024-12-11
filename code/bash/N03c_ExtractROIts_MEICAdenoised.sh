# 10/22/2024 - Javier Gonzalez-Castillo
#
# This script will generate MEICA denoised versions of the data with
# the three different censoring schemes provided by 3dTproject
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

echo "++ Denoising data..."
echo "===================="
for INTERP_MODE in NTRP ZERO KILL
do
   echo " + Working with interpolation mode ${INTERP_MODE}"
   3dTproject -overwrite                                                 \
              -polort 0                                                  \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD                      \
              -censor censor_${SBJ}_combined_2.1D                        \
              -cenmode ${INTERP_MODE}                                    \
              -ort X.nocensor.xmat.1D                                    \
              -prefix errts.${SBJ}.r01.OC_MEICA.tproject_${INTERP_MODE}  \
              -mask mask_tedana_at_least_one_echo.nii.gz
done

echo " + Working with interpolation mode ALL"
3dTproject -overwrite                                         \
              -polort 0                                       \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD          \
              -ort X.nocensor.xmat.1D                         \
              -prefix errts.${SBJ}.r01.OC_MEICA.tproject_ALL  \
              -mask mask_tedana_at_least_one_echo.nii.gz

echo "++ Extracting Timeseries and FC matrices"
echo "========================================"
for INTERP_MODE in NTRP ZERO KILL ALL
do
echo " + Extracting ROI Timeseries and connectivity for [${INTERP_MODE}]"
    3dNetCorr -overwrite                                                               \
              -mask mask_tedana_at_least_one_echo.nii.gz                               \
              -in_rois ${ATLAS_PATH}                                                   \
              -inset  errts.${SBJ}.r01.OC_MEICA.tproject_${INTERP_MODE}+tlrc          \
              -prefix errts.${SBJ}.r01.OC_MEICA.tproject_${INTERP_MODE}.${ATLAS_NAME}

    3dROIstats -quiet              \
               -mask ${ATLAS_PATH} \
                     errts.${SBJ}.r01.OC_MEICA.tproject_${INTERP_MODE}+tlrc > errts.${SBJ}.r01.OC_MEICA.tproject_${INTERP_MODE}.${ATLAS_NAME}_000.netts
done


echo "==============================="
echo "++ Script Finished successfully"
echo "==============================="