# JGC 06/06/2025: Compute Thermal noise outside the brain
set -e 

ml afni

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024'   # Project directory: includes Scripts, Freesurfer and PrcsData folders

PRCSDATA_DIR=`echo ${PRJDIR}/prcs_data`
DOWNLOAD_DIR=`echo ${PRJDIR}/openeuro/des003592-download/`
SUBJECTS_DIR=`echo ${PRJDIR}/freesurfer/`
SCRIPTS_DIR=`echo ${PRJDIR}/me_staticfc/code/bash`
AFNI_PROC_OUT_DIR=`echo ${SCRIPTS_DIR}/S04_Afni_Preproc_fMRI_${SES}.NORDIC`
RESOURCES_DIR='/data/SFIMJGC_HCP7T/BCBL2024/resources/'
USERNAME=`whoami`
SWARM_PATH=`echo ${PRJDIR}/swarm.${USERNAME}/S04_Afni_Preproc_fMRI_${SES}.NORDIC.SWARM.sh`
LOGS_DIR=`echo ${PRJDIR}/logs.${USERNAME}/S04_Afni_Preproc_fMRI_${SES}.NORDIC.logs`
sessions=(ses-1 ses-2)
echo "++ Orig Data Folder  : ${PRCSDATA_DIR}"
echo "++ Scripts Folder    : ${SCRIPTS_PATH}"
echo "++ Swarm Folder      : ${SWARM_PATH}"
echo "++ Logs Folder       : ${LOGS_DIR}"
echo "++ Freesurfer Folder : ${SUBJECTS_DIR}"
echo "++ Afni Proc Out Dir : ${AFNI_PROC_OUT_DIR}"


# Generate Signal-Free mask outside the brain for computation of Thermal Noise
# ============================================================================
for SES in ${sessions[@]}
do
   PRE_NORDIC_DIR=`echo ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/`
   POST_NORDIC_DIR=`echo ${PRCSDATA_DIR}/${SBJ}/D03_NORDIC/`
  
   # Compute Mean signal level per voxel in each echo (pre-NORDIC) 
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e01.pre_nordic.stats.nii.gz ${PRE_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-1_bold.nii.gz
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e02.pre_nordic.stats.nii.gz ${PRE_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-2_bold.nii.gz
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e03.pre_nordic.stats.nii.gz ${PRE_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-3_bold.nii.gz

   # Compute Mean signal level per voxel in each echo (post-NORDIC) 
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e01.post_nordic.stats.nii.gz ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-1_bold.NORDIC.nii.gz
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e02.post_nordic.stats.nii.gz ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-2_bold.NORDIC.nii.gz
   3dTstat -overwrite -mean -zcount -prefix ${POST_NORDIC_DIR}/rm.${SES}.e03.post_nordic.stats.nii.gz ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-3_bold.NORDIC.nii.gz

   # We will represent each voxel as 3 numbers, which are the mean values.
   # Vectors with a very small norm (e.g., 1) can be considered free of signal
   # NOTE: We will also make sense we discard weird voxels that have been set to zero too often (not real data in there)
   3dcalc -overwrite -a ${POST_NORDIC_DIR}/rm.${SES}.e01.pre_nordic.stats.nii.gz['Mean'] -b ${POST_NORDIC_DIR}/rm.${SES}.e02.pre_nordic.stats.nii.gz['Mean'] -c ${POST_NORDIC_DIR}/rm.${SES}.e03.pre_nordic.stats.nii.gz['Mean'] -expr 'sqrt((a-b)^2 + (a-c)^2 + (b-c)^2)' -prefix ${POST_NORDIC_DIR}/rm.${SES}.pre_nordic.euc_dist_means.nii.gz
   3dcalc -overwrite -a ${POST_NORDIC_DIR}/rm.${SES}.e01.post_nordic.stats.nii.gz['Mean'] -b ${POST_NORDIC_DIR}/rm.${SES}.e02.post_nordic.stats.nii.gz['Mean'] -c ${POST_NORDIC_DIR}/rm.${SES}.e03.post_nordic.stats.nii.gz['Mean'] -expr 'sqrt((a-b)^2 + (a-c)^2 + (b-c)^2)' -prefix ${POST_NORDIC_DIR}/rm.${SES}.post_nordic.euc_dist_means.nii.gz
   3dcalc -overwrite -a ${POST_NORDIC_DIR}/rm.${SES}.pre_nordic.euc_dist_means.nii.gz   \
                     -b ${POST_NORDIC_DIR}/rm.${SES}.post_nordic.euc_dist_means.nii.gz  \
                     -c ${POST_NORDIC_DIR}/rm.${SES}.e01.pre_nordic.stats.nii.gz['ZeroCount']   \
                     -d ${POST_NORDIC_DIR}/rm.${SES}.e02.pre_nordic.stats.nii.gz['ZeroCount']   \
                     -e ${POST_NORDIC_DIR}/rm.${SES}.e03.pre_nordic.stats.nii.gz['ZeroCount']   \
                     -f ${POST_NORDIC_DIR}/rm.${SES}.e01.post_nordic.stats.nii.gz['ZeroCount']  \
                     -g ${POST_NORDIC_DIR}/rm.${SES}.e02.post_nordic.stats.nii.gz['ZeroCount']  \
                     -h ${POST_NORDIC_DIR}/rm.${SES}.e03.post_nordic.stats.nii.gz['ZeroCount']  \
                     -expr 'isnegative(a-2)*isnegative(b-2)*isnegative(c-20)*isnegative(d-20)*isnegative(e-20)*isnegative(f-20)*isnegative(g-20)*isnegative(h-20)' -prefix ${POST_NORDIC_DIR}/rm.${SES}.post_nordic.thermal_noise_mask.nii.gz
done

# Generate final mask for computation of thermal noise
# ====================================================
3dcalc -overwrite -a ${POST_NORDIC_DIR}/rm.ses-1.post_nordic.thermal_noise_mask.nii.gz \
                  -b ${POST_NORDIC_DIR}/rm.ses-2.post_nordic.thermal_noise_mask.nii.gz  \
                  -expr 'a*b' \
                  -prefix ${POST_NORDIC_DIR}/${SBJ}.thermal_noise_mask.nii.gz

NUM_VOXELS_IN_MASK=`3dROIstats -nzvoxels -nomeanout -quiet -mask ${POST_NORDIC_DIR}/${SBJ}.thermal_noise_mask.nii.gz ${POST_NORDIC_DIR}/${SBJ}.thermal_noise_mask.nii.gz`
echo "${SBJ} ${NUM_VOXELS_IN_MASK}" > ${POST_NORDIC_DIR}/${SBJ}.thermal_noise_mask.size.txt
echo "===============================> ${SBJ} has ${NUM_VOXELS_IN_MASK} voxels in thermal noise mask." 
rm ${POST_NORDIC_DIR}/rm.*

# Compute Thermal Noise using the mask just generated
# ===================================================
for scenario in ORIG NORDIC
do
   for SES in ${sessions[@]}
   do
      PRE_NORDIC_DIR=`echo ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/`
      POST_NORDIC_DIR=`echo ${PRCSDATA_DIR}/${SBJ}/D03_NORDIC/`
      for e in 1 2 3
      do
          if [ "${scenario}" == NORDIC ]; then
             INPUT_FILE=`echo ${PRE_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-${e}_bold.nii.gz`
             OUTPUT_FILE=`echo ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-${e}_bold.NORDIC_off.ThermalNoise.txt`
          else
             INPUT_FILE=`echo ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-${e}_bold.NORDIC.nii.gz`
             OUTPUT_FILE=`echo ${POST_NORDIC_DIR}/${SBJ}_${SES}_task-rest_echo-${e}_bold.NORDIC_on.ThermalNoise.txt`
          fi
          3dROIstats -sigma -nomeanout -quiet -mask ${POST_NORDIC_DIR}/${SBJ}.thermal_noise_mask.nii.gz ${INPUT_FILE} > ${OUTPUT_FILE}
     done
  done
done
