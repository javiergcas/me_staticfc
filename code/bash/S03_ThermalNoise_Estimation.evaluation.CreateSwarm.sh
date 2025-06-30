set -e

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024'
DOWNLOAD_DIR=`echo ${PRJDIR}/openeuro/des003592-download/`
SUBJECTS_DIR=`echo ${PRJDIR}/freesurfer/`
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data/`
SCRIPTS_DIR=`echo ${PRJDIR}/me_staticfc/code/bash`
USERNAME=`whoami`

SWARM_PATH=`echo ${PRJDIR}/swarm.${USERNAME}/S03_ThermalNoise_Estimation.evaluation.SWARM.sh`
LOGS_DIR=`echo ${PRJDIR}/logs.${USERNAME}/S03_ThermalNoise_Estimation.evaluation.logs`
subjects=(`find ${SUBJECTS_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
#echo "++ Subjects          : ${subjects[@]}"
echo "++ Orig Data Folder  : ${DOWNLOAD_DIR}"
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
   echo "export SBJ=${sbj} DOWNLOAD_DIR=${DOWNLOAD_DIR} DATASET=evaluation; sh ${SCRIPTS_DIR}/S03_ThermalNoise_Estimation.sh" >> ${SWARM_PATH}
done

echo "++ INFO: Script finished correctly."
