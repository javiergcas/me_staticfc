set -e

module load afni

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024'   # Project directory: includes Scripts, Freesurfer and PrcsData folders


SES=ses-1
PRCSDATA_DIR=`echo ${PRJDIR}/prcs_data`
DOWNLOAD_DIR=`echo ${PRJDIR}/openeuro/des003592-download`
SUBJECTS_DIR=`echo ${PRJDIR}/freesurfer/`
SCRIPTS_DIR=`echo ${PRJDIR}/me_staticfc/code/bash`
AFNI_PROC_OUT_DIR=`echo ${SCRIPTS_DIR}/S02_Afni_Preproc_fMRI_${SES}`
RESOURCES_DIR='/data/SFIMJGC_HCP7T/BCBL2024/resources/'
USERNAME=`whoami`
SWARM_PATH=`echo ${PRJDIR}/swarm.${USERNAME}/S02_Afni_Preproc_fMRI_${SES}.SWARM.sh`
LOGS_DIR=`echo ${PRJDIR}/logs.${USERNAME}/S02_Afni_Preproc_fMRI_${SES}.logs`
subjects=(`find ${SUBJECTS_DIR} -name "sub-*" -type d | tr -s '\n' ' '`)
num_subjects=`echo ${#subjects[@]}`
echo "++ Orig Data Folder  : ${PRCSDATA_DIR}"
echo "++ Scripts Folder    : ${SCRIPTS_PATH}"
echo "++ Swarm Folder      : ${SWARM_PATH}"
echo "++ Logs Folder       : ${LOGS_DIR}"
echo "++ Freesurfer Folder : ${SUBJECTS_DIR}"
echo "++ Afni Proc Out Dir : ${AFNI_PROC_OUT_DIR}"

# Initialize Swarm File
# ---------------------
echo "#Creation Time: `date`" > ${SWARM_PATH}
echo "#swarm -f ${SWARM_PATH} -g 32 -t 32 --time 48:00:00 --module afni --logdir ${LOGS_DIR} --sbatch \"--export AFNI_COMPRESSOR=GZIP\"" >> ${SWARM_PATH}

# Create log directory if needed (for swarm files)
# ------------------------------------------------
if [ ! -d ${LOGS_DIR} ]; then 
   mkdir ${LOGS_DIR}
fi

# Create directory for all fMRI data processing files per subject if needed
# -------------------------------------------------------------------------
if [ ! -d ${AFNI_PROC_OUT_DIR} ]; then 
   mkdir ${AFNI_PROC_OUT_DIR}
fi

# Copy and process all fMRI data
# ------------------------------
for sbj_path in ${subjects[@]}
do
    SBJ=`basename ${sbj_path}`
    ANAT_PROC_DIR=`echo ${PRJDIR}/prcs_data/${SBJ}/D01_Anatomical`
    if [ ! -f ${ANAT_PROC_DIR}/anatQQ.${SBJ}.nii.gz ]; then
       echo "${SBJ} ${SES}" >> ${RESOURCES_DIR}/S02_NotAvailable.txt
    else
       echo "${SBJ} ${SES}" >> ${RESOURCES_DIR}/S02_WillTry.txt
       FMRI_ORIG_DIR=`echo ${DOWNLOAD_DATA}/${SBJ}/${SES}/func`
       OUT_DIR=`echo ${PRJDIR}/prcs_data/${SBJ}/D02_Preproc_fMRI_${SES}`

       SITE=`grep "${SBJ}\b" ${DOWNLOAD_DIR}/participants.tsv | awk -F '\t' '{print $3}'`
       if [[ "${SITE}" == "1" ]]; then ECHOTIMES="13.7 30 47"; else ECHOTIMES="14 29.96 45.92"; fi
       echo "${SBJ} Site=${SITE} --> Echoes=${ECHOTIMES}"
       afni_proc.py                                                                                          \
                -subj_id ${SBJ}                                                                              \
                -blocks despike tshift align tlrc volreg mask combine scale regress                          \
                -radial_correlate_blocks tcat volreg                                                         \
                -copy_anat ${ANAT_PROC_DIR}/anatSS.${SBJ}.nii.gz                                             \
                -anat_has_skull no                                                                           \
                -anat_follower anat_w_skull anat ${ANAT_PROC_DIR}/anatUAC.${SBJ}.nii.gz                      \
                -anat_follower_ROI aaseg  anat ${SUBJECTS_DIR}/${SBJ}/SUMA/aparc.a2009s+aseg.nii.gz          \
                -anat_follower_ROI aeseg  epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/aparc.a2009s+aseg.nii.gz          \
                -anat_follower_ROI FSvent epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/fs_ap_latvent.nii.gz              \
                -anat_follower_ROI FSWe   epi  ${SUBJECTS_DIR}/${SBJ}/SUMA/fs_ap_wm.nii.gz                   \
                -anat_follower_erode FSvent FSWe                                                             \
                -tcat_remove_first_trs 3                                                                     \
                -tshift_interp -wsinc9                                                                       \
                -dsets_me_run ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-1_bold.nii.gz  \
                              ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-2_bold.nii.gz  \
                              ${DOWNLOAD_DIR}/${SBJ}/${SES}/func/${SBJ}_${SES}_task-rest_echo-3_bold.nii.gz  \
                -echo_times ${ECHOTIMES}                                                                     \
                -combine_method m_tedana                                                                     \
                -combine_opts_tedana --verbose                                                               \
                -align_unifize_epi local                                                                     \
                -align_opts_aea -cost lpc+ZZ -giant_move -check_flip                                         \
                -tlrc_base MNI152_2009_template_SSW.nii.gz                                                   \
                -tlrc_NL_warp                                                                                \
                -tlrc_NL_warped_dsets ${ANAT_PROC_DIR}/anatQQ.${SBJ}.nii.gz                                  \
                      ${ANAT_PROC_DIR}/anatQQ.${SBJ}.aff12.1D                                                \
                      ${ANAT_PROC_DIR}/anatQQ.${SBJ}_WARP.nii.gz                                             \
                -volreg_align_to MIN_OUTLIER                                                                 \
                -volreg_align_e2a                                                                            \
                -volreg_tlrc_warp                                                                            \
                -volreg_warp_dxyz 3                                                                          \
                -volreg_warp_final_interp  wsinc5                                                            \
                -volreg_compute_tsnr       yes                                                               \
                -mask_epi_anat yes                                                                           \
                -regress_opts_3dD -jobs 32                                                                   \
                -regress_motion_per_run                                                                      \
                -regress_ROI_PC FSvent 3                                                                     \
                -regress_ROI_PC_per_run FSvent                                                               \
                -regress_make_corr_vols aeseg FSvent                                                         \
                -regress_anaticor_fast                                                                       \
                -regress_anaticor_label FSWe                                                                 \
                -regress_censor_motion 0.2                                                                   \
                -regress_censor_outliers 0.05                                                                \
                -regress_apply_mot_types demean deriv                                                        \
                -regress_bandpass 0.01 0.2                                                                   \
                -regress_polort 5                                                                            \
                -regress_run_clustsim no                                                                     \
                -html_review_style pythonic                                                                  \
                -out_dir ${OUT_DIR}                                                                          \
                -script ${AFNI_PROC_OUT_DIR}/S02_Afni_Preproc_fMRI.${SBJ}_${SES}.sh                          \
                -regress_compute_tsnr yes                                                                    \
                -regress_make_cbucket yes                                                                    \
                -scr_overwrite

       # Add line for this subject to the Swarm file
       echo "module load afni; source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate tedana_2024a; tcsh -xef ${AFNI_PROC_OUT_DIR}/S02_Afni_Preproc_fMRI.${SBJ}_${SES}.sh 2>&1 | tee ${AFNI_PROC_OUT_DIR}/output.S02_Afni_Preproc_fMRI.${SBJ}_${SES}.txt" >> ${SWARM_PATH}
    fi
done

echo "============================================================================================"
echo " === SWARM FILE IN: ${SWARM_PATH}"
echo "============================================================================================"
