# 12/31/2024 - Javier Gonzalez-Castillo
#
# This script will perform the rapidtide based denoising

export OMP_NUM_THREADS=32
export AFNI_COMPRESSOR=GZIP

set -e

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024/'
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data`

SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`
FMRI_DATA_DIR=`echo ${SBJ_DIR}/D02_Preproc_fMRI_${SES}`
#ATLAS_PATH=`echo ${ATLASES_DIR}/${ATLAS_NAME}/${ATLAS_NAME}.nii.gz`

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

# Create a Grey Matter ribbon mask
# ================================
echo "++ Creating GM Ribbon mask based on freesurfer"
echo "++ -------------------------------------------"
for hemi in rh lh
do
   3dNwarpApply -overwrite \
                -source ../../../freesurfer/${SBJ}/SUMA/${hemi}.ribbon.nii.gz \
                -master pb03.${SBJ}.r01.e02.volreg+tlrc. \
                -ainterp NN \
                -nwarp anatQQ.${SBJ}_WARP.nii.gz anatQQ.${SBJ}.aff12.1D \
                -prefix follow_ROI_${hemi}_ribbon.nii.gz
done

3dcalc -overwrite -a follow_ROI_lh_ribbon.nii.gz -b follow_ROI_rh_ribbon.nii.gz -expr 'a+b' -prefix follow_ROI_GM.nii.gz

rm follow_ROI_lh_ribbon.nii.gz follow_ROI_rh_ribbon.nii.gz

# Bring WM mask to EPI space and grid
echo "++ Move Freesurfer WM mask to EPI grid in MNI space"
echo "++ ------------------------------------------------"
3dNwarpApply -overwrite \
             -source ../../../freesurfer/${SBJ}/SUMA/wm.seg.nii.gz \
             -master pb03.${SBJ}.r01.e02.volreg+tlrc. \
             -ainterp NN \
             -nwarp anatQQ.${SBJ}_WARP.nii.gz anatQQ.${SBJ}.aff12.1D \
             -prefix follow_ROI_WM.nii.gz

# Run rapidtide
echo "++ Convert rest of inputs from BRIK/HEAD to nii.gz"
echo "++ -----------------------------------------------"
3dcopy -overwrite mask_group+tlrc mask_group.nii.gz
3dcopy -overwrite pb05.${SBJ}.r01.scale+tlrc pb05.${SBJ}.r01.scale.nii.gz

echo "++ Run rapidtide for this scan"
echo "++ ---------------------------"
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate rapidtide_2024

rapidtide ./pb05.${SBJ}.r01.scale.nii.gz \
          ./pb05.${SBJ}.temp_rapidtide \
          --denoising \
          --brainmask mask_group.nii.gz \
          --graymattermask follow_ROI_GM.nii.gz \
          --whitemattermask follow_ROI_WM.nii.gz \
          --nprocs 32 \
          --detrendorder 5 \
          --motionfile ./motion_demean.1D \
          --filterband lfo \
          --mklthreads 32 \
          --outputlevel max \
          --savelags
          
echo " +  `pwd`"
echo "++ Clean up headers on rapidtide outputs"
echo "++ -------------------------------------"
nifti_tool -strip_extras -overwrite -infiles pb05.${SBJ}.temp_rapidtide_desc*nii.gz

mv pb05.sub-01.temp_rapidtide_desc-EV_timeseries.json      pb05.sub-01.rtide_desc-EV_timeseries.json
mv pb05.sub-01.temp_rapidtide_desc-lfofilterEV_bold.json   pb05.sub-01.rtide_desc-lfofilterEV_bold.json
mv pb05.sub-01.temp_rapidtide_desc-EV_timeseries.tsv.gz    pb05.sub-01.rtide_desc-EV_timeseries.tsv.gz
mv pb05.sub-01.temp_rapidtide_desc-lfofilterEV_bold.nii.gz pb05.sub-01.rtide_desc-lfofilterEV_bold.nii.gz

echo "++ Remove redundant files"
echo "++ ----------------------"
rm mask_group.nii.gz
rm pb05.${SBJ}.r01.scale.nii.gz
rm pb05.sub-01.temp_rapidtide_desc*


echo "++ Denoising each echo separately (using Rapidtide regressor)"
echo "++ ----------------------------------------------------------"
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
echo "++ =================================="
echo "++ == Script finished successfully =="
echo "++ =================================="

