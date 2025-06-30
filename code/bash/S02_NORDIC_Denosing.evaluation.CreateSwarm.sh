set -e


PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024'   # Project directory: includes Scripts, Freesurfer and PrcsData folders
PRCSDATA_DIR=`echo ${PRJDIR}/prcs_data`
DOWNLOAD_DIR=`echo ${PRJDIR}/openeuro/des003592-download`
SUBJECTS_DIR=`echo ${PRJDIR}/freesurfer/`
SCRIPTS_DIR=`echo ${PRJDIR}/me_staticfc/code/matlab`
USERNAME=`whoami`
SWARM_PATH=`echo ${PRJDIR}/swarm.${USERNAME}/S02_NORDIC_Denosing.SWARM.sh`
LOGS_DIR=`echo ${PRJDIR}/logs.${USERNAME}/S02_NORDIC_Denosing.logs`
subjects=(`find ${SUBJECTS_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
sessions=(ses-1 ses-2)
echo "++ Orig Data Folder  : ${PRCSDATA_DIR}"
echo "++ Scripts Folder    : ${SCRIPTS_DIR}"
echo "++ Swarm Folder      : ${SWARM_PATH}"
echo "++ Logs Folder       : ${LOGS_DIR}"
echo "++ Freesurfer Folder : ${SUBJECTS_DIR}"

# Initialize Swarm File
# ---------------------
echo "#Creation Time: `date`" > ${SWARM_PATH}
echo "#swarm -f ${SWARM_PATH} -g 32 -t 4 -b 5 --time 00:10:00 --module matlab --partition quick,norm --logdir ${LOGS_DIR}" >> ${SWARM_PATH}

# Create log directory if needed (for swarm files)
# ------------------------------------------------
if [ ! -d ${LOGS_DIR} ]; then 
   mkdir ${LOGS_DIR}
fi

# Copy and process all fMRI data
# ------------------------------
for sbj_path in ${subjects[@]}
do
   SBJ=`basename ${sbj_path}`
   for SES in ${sessions[@]}
   do
       INPUT_FILE=`echo ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-1_bold.nii.gz`
       if [ -f ${INPUT_FILE} ]; then
          NORDIC_OUT_DIR=`echo ${PRCSDATA_DIR}/${SBJ}/D03_NORDIC`
          if [ ! -d ${NORDIC_OUT_DIR} ]; then
             mkdir -p ${NORDIC_OUT_DIR}
          fi
          echo "sh ${SCRIPTS_DIR}/run_NORDIC.sh ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-1_bold.nii.gz ${NORDIC_OUT_DIR}/${SBJ}_${SES}_task-rest_echo-1_bold.NORDIC" >> ${SWARM_PATH}
          echo "sh ${SCRIPTS_DIR}/run_NORDIC.sh ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-2_bold.nii.gz ${NORDIC_OUT_DIR}/${SBJ}_${SES}_task-rest_echo-2_bold.NORDIC" >> ${SWARM_PATH}
          echo "sh ${SCRIPTS_DIR}/run_NORDIC.sh ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-3_bold.nii.gz ${NORDIC_OUT_DIR}/${SBJ}_${SES}_task-rest_echo-3_bold.NORDIC" >> ${SWARM_PATH}
       else
          echo "++ WARNING: ${INPUT_FILE} is missing"
       fi
   done 
done

echo "++ INFO: Swarm File [${SWARM_PATH}] has `cat ${SWARM_PATH} | wc -l` entries"
echo "++ ======================================================="
head -n 5 ${SWARM_PATH}
echo "++ ======================================================="
