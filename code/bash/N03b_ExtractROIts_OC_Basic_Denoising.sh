# 10/22/2024 - Javier Gonzalez-Castillo
#
# This script will generate MEICA denoised versions of the data with
# the three different censoring schemes provided by 3dTproject
#

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

# =======================================================================
# NOTE: 
# This is likely not needed for this project, but we leave it for now.
# =======================================================================
echo "++ Applying basic denoising to the OC timeseries"
echo "++ ---------------------------------------------"
for INTERP_MODE in NTRP ZERO KILL
do
   echo " + Working with interpolation mode ${INTERP_MODE}"
   3dTproject -overwrite                                                 \
              -polort 0                                                  \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD                     \
              -censor censor_${SBJ}_combined_2.1D                        \
              -cenmode ${INTERP_MODE}                                    \
              -ort X.nocensor.xmat.1D                                    \
              -prefix errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}        \
              -mask mask_tedana_at_least_one_echo.nii.gz
done

echo " + Working with interpolation mode ALL"
3dTproject -overwrite                                         \
              -polort 0                                       \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD          \
              -ort X.nocensor.xmat.1D                         \
              -prefix errts.${SBJ}.r01.OC.tproject_ALL        \
              -mask mask_tedana_at_least_one_echo.nii.gz

echo "++ Global Signal regression on OC timeseries"
echo "++ -----------------------------------------"
# Global regression with no detrending prior to computing GS
3dROIstats -quiet \
           -mask mask_tedana_at_least_one_echo.nii.gz \
                 pb05.${SBJ}.r01.scale+tlrc.HEAD | awk '{print $1}' > pb05.${SBJ}.r01.scale.GSasis.1D

# Alternative GS regression computing GS post detrending           
3dDetrend -overwrite \
          -polort 5 \
          -prefix pb05.${SBJ}.r01.scale.dt5 \
                  pb05.${SBJ}.r01.scale+tlrc
3dROIstats -quiet \
               -mask mask_tedana_at_least_one_echo.nii.gz \
               pb05.${SBJ}.r01.scale.dt5+tlrc.HEAD | awk '{print $1}' > pb05.${SBJ}.r01.scale.GSdt5.1D
rm pb05.${SBJ}.r01.scale.dt5+tlrc.*

for INTERP_MODE in NTRP ZERO KILL
do
   echo " + Working with interpolation mode ${INTERP_MODE}"
   3dTproject -overwrite                                                 \
              -polort 0                                                  \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD                     \
              -censor censor_${SBJ}_combined_2.1D                        \
              -cenmode ${INTERP_MODE}                                    \
              -ort X.nocensor.xmat.1D                                    \
              -ort pb05.${SBJ}.r01.scale.GSasis.1D                       \
              -prefix errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}_GSasis \
              -mask mask_tedana_at_least_one_echo.nii.gz
    3dTproject -overwrite                                                \
              -polort 0                                                  \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD                     \
              -censor censor_${SBJ}_combined_2.1D                        \
              -cenmode ${INTERP_MODE}                                    \
              -ort X.nocensor.xmat.1D                                    \
              -ort pb05.${SBJ}.r01.scale.GSdt5.1D                        \
              -prefix errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}_GSdt5  \
              -mask mask_tedana_at_least_one_echo.nii.gz
done

echo " + Working with interpolation mode ALL"
3dTproject -overwrite                                           \
              -polort 0                                         \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD            \
              -ort X.nocensor.xmat.1D                           \
              -ort pb05.${SBJ}.r01.scale.GSasis.1D              \
              -prefix errts.${SBJ}.r01.OC.tproject_ALL_GSasis   \
              -mask mask_tedana_at_least_one_echo.nii.gz
3dTproject -overwrite                                           \
              -polort 0                                         \
              -input pb05.${SBJ}.r01.scale+tlrc.HEAD            \
              -ort X.nocensor.xmat.1D                           \
              -ort pb05.${SBJ}.r01.scale.GSdt5.1D               \
              -prefix errts.${SBJ}.r01.OC.tproject_ALL_GSdt5    \
              -mask mask_tedana_at_least_one_echo.nii.gz


echo "++ Extracting Timeseries and FC matrices"
echo "========================================"
for INTERP_MODE in NTRP ZERO KILL ALL NTRP_GSasis ZERO_GSasis KILL_GSasis ALL_GSasis NTRP_GSdt5 ZERO_GSdt5 KILL_GSdt5 ALL_GSdt5
do
echo " + Extracting ROI Timeseries and connectivity for [${INTERP_MODE}]"
    3dNetCorr -overwrite                                                               \
              -mask mask_tedana_at_least_one_echo.nii.gz                               \
              -in_rois ${ATLAS_PATH}                                                   \
              -inset  errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}+tlrc                 \
              -prefix errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}.${ATLAS_NAME}

    3dROIstats -quiet                                                                  \
               -mask ${ATLAS_PATH}                                                     \
                     errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}+tlrc > errts.${SBJ}.r01.OC.tproject_${INTERP_MODE}.${ATLAS_NAME}_000.netts
done

# Once we extract ROI timeseries and FC matrices, it is very unlike that we will need the data in this form --> to save space --> remove
rm errts.${SBJ}.r01.OC.tproject_NTRP+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_ZERO+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_ALL_GSdt5+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_ALL_GSasis+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_KILL_GSdt5+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_KILL_GSasis+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_NTRP_GSdt5+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_NTRP_GSasis+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_ZERO_GSdt5+tlrc.*
rm errts.${SBJ}.r01.OC.tproject_ZERO_GSasis+tlrc.*

echo "==============================="
echo "++ Script Finished successfully"
echo "==============================="
