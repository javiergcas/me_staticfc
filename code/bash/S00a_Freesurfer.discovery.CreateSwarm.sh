set -e

ORIG_DATA_DIR='/data/SFIMJGC_HCP7T/BCBL2024/openeuro/des003592-download/'
SUBJECTS_DIR='/data/SFIMJGC_HCP7T/BCBL2024/freesurfer/'
PRCS_DATA_DIR='/data/SFIMJGC_HCP7T/BCBL2024/prcs_data/'
USERNAME=`whoami`
SWARM_PATH=`echo /data/SFIMJGC_HCP7T/BCBL2024/swarm.${USERNAME}/S00a_Freesurfer.SWARM.sh`
LOGS_DIR=`echo /data/SFIMJGC_HCP7T/BCBL2024/logs.${USERNAME}/S00a_Freesurfer.logs`
subjects=(`find ${ORIG_DATA_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
echo "++ Subjects          : ${subjects[@]}"
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

# Create Fresurfer directory if needed
# ------------------------------------
if [ ! -d ${SUBJECTS_DIR} ]; then
   mkdir ${SUBJECTS_DIR}
fi

# Write top comment in Swarm file 
# -------------------------------
echo "#Creation Date: `date`" > ${SWARM_PATH}
echo "#swarm -f ${SWARM_PATH} -g 24 -t 24 --time 12:00:00 --logdir ${LOGS_DIR} --module afni,freesurfer --sbatch \"--export SUBJECTS_DIR=${SUBJECTS_DIR}\"" >> ${SWARM_PATH}
# Write one entry per subject in Swarm file
for sbj_path in ${subjects[@]}
do
    sbj=`basename ${sbj_path}`
    out_path=`echo ${PRCS_DATA_DIR}/${sbj}/D00_PreFreesurfer`
    if [ ! -d ${out_path} ]; then
       mkdir ${out_path}
    fi
    echo "3dUnifize -overwrite -prefix ${out_path}/${sbj}_ses-1_T1w.unifize.nii.gz ${sbj_path}/ses-1/anat/${sbj}_ses-1_T1w.nii.gz; recon-all -all -subject ${sbj} -i ${out_path}/${sbj}_ses-1_T1w.unifize.nii.gz" >> ${SWARM_PATH} 
done
echo "++ INFO: Script finished correctly."
