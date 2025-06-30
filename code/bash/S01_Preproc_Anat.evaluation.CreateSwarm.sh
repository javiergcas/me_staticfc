set -e

ORIG_DATA_DIR='/data/SFIMJGC_HCP7T/BCBL2024/openeuro/des003592-download/'
SUBJECTS_DIR='/data/SFIMJGC_HCP7T/BCBL2024/freesurfer/'
PRCS_DATA_DIR='/data/SFIMJGC_HCP7T/BCBL2024/prcs_data/'
RESOURCES_DIR='/data/SFIMJGC_HCP7T/BCBL2024/resources/'
SCRIPTS_DIR='/data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/bash/'
USERNAME=`whoami`
SWARM_PATH=`echo /data/SFIMJGC_HCP7T/BCBL2024/swarm.${USERNAME}/S01_Preproc_Anat.SWARM.sh`
LOGS_DIR=`echo /data/SFIMJGC_HCP7T/BCBL2024/logs.${USERNAME}/S01_Preproc_Anat.logs`
subjects=(`find ${SUBJECTS_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
#echo "++ Subjects          : ${subjects[@]}"
echo "++ Orig Data Folder  : ${ORIG_DATA_DIR}"
echo "++ Swarm Folder      : ${SWARM_PATH}"
echo "++ Logs Folder       : ${LOGS_DIR}"
echo "++ Freesurfer Folder : ${SUBJECTS_DIR}"
echo "++ Number of subjects: ${num_subjects}"


# Create log directory if needed
# ------------------------------
if [ ! -d ${LOGS_DIR} ]; then
   mkdir ${LOGS_DIR}
fi

# create Pre directory if needed
# ------------------------------
if [ ! -d ${PRCS_DATA_DIR} ]; then
   mkdir ${PRCS_DATA_DIR}
fi



# Write top comment in Swarm file 
# -------------------------------
echo "#Creation Date: `date`" > ${SWARM_PATH}
echo "#swarm -f ${SWARM_PATH} -g 32 -t 32 --partition quick,norm --module afni --logdir ${LOGS_DIR} --sbatch \"--export AFNI_COMPRESSOR=GZIP\"" > ${SWARM_PATH}

for sbj_path in ${subjects[@]}
do
   sbj=`basename ${sbj_path}`
   if [ ! -d ${SUBJECTS_DIR}/${sbj}/SUMA ]; then 
      echo "${sbj}" >> ${RESOURCES_DIR}/S01_NotAvailable.txt
   else
      if [ -z "$(ls -A ${SUBJECTS_DIR}/${sbj}/SUMA)" ]; then
        echo "${sbj}" >> ${RESOURCES_DIR}/S01_NotAvailable.txt
      else
        echo "${sbj}" >> ${RESOURCES_DIR}/S01_WillTry.txt
        echo "export SBJ=${sbj}; sh ${SCRIPTS_DIR}/S01_Preproc_Anat.sh" >> ${SWARM_PATH}
      fi
   fi
done

echo "++ INFO: Script finished correctly."
