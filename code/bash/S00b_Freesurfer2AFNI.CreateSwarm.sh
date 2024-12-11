set -e

ORIG_DATA_DIR='/data/SFIMJGC_HCP7T/BCBL2024/openeuro/des003592-download/'
SUBJECTS_DIR='/data/SFIMJGC_HCP7T/BCBL2024/freesurfer/'
RESOURCES_DIR='/data/SFIMJGC_HCP7T/BCBL2024/resources/'
USERNAME=`whoami`
SWARM_PATH=`echo /data/SFIMJGC_HCP7T/BCBL2024/swarm.${USERNAME}/S00b_Freesurfer2AFNI.SWARM.sh`
LOGS_DIR=`echo /data/SFIMJGC_HCP7T/BCBL2024/logs.${USERNAME}/S00b_Freesurfer2AFNI.logs`
subjects=(`find ${SUBJECTS_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
#echo "++ Subjects          : ${subjects[@]}"
echo "++ Orig Data Folder  : ${ORIG_DATA_DIR}"
echo "++ Swarm Folder      : ${SWARM_PATH}"
echo "++ Logs Folder       : ${LOGS_DIR}"
echo "++ Freesurfer Folder : ${SUBJECTS_DIR}"
echo "++ Number of subjects: ${num_subjects}"

# Create log directory if needed
if [ ! -d ${LOGS_DIR} ]; then
   mkdir ${LOGS_DIR}
fi

# Write top comment in CreateSwarm File
echo "#Creation Date: `date`" > ${SWARM_PATH}
echo "#swarm -f ${SWARM_PATH} -g 24 -t 24 --partition quick,norm --logdir ${LOGS_DIR}  --module afni,freesurfer" >> ${SWARM_PATH}

# Write one entry per subject in CreateSwarm File
for sbj_path in ${subjects[@]}
do
   sbj="$(basename ${sbj_path})"
   if [ ! -f ${sbj_path}/surf/lh.pial ]; then
           echo "$sbj" >> ${RESOURCES_DIR}/S00b_NotAvailable.txt
   else 
      echo "$sbj" >> ${RESOURCES_DIR}/S00b_WillTry.txt
      if [ ! -d ${sbj_path}/SUMA ]; then
           echo "@SUMA_Make_Spec_FS -sid ${sbj}  -NIFTI -fspath ${sbj_path}" >> ${SWARM_PATH} 
      fi
   fi
done

echo "++ INFO: Script finished correctly."
