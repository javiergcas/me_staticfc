# 07/02/2025 - Javier Gonzalez-Castillo
# This script will do the steps beyond afni_proc needed to complete the
# TEDANA with robustICA.

export OMP_NUM_THREADS=32
set -e
ml afni
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate tedana_2025a
PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024/'
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data`
SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`
FMRI_DATA_DIR=`echo ${SBJ_DIR}/D03_Preproc_${SES}_NORDIC-${NORDIC}`

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

if [ -d tedana_robustica ]; then
   echo "++ ============================================================= ++"
   echo "++ WARNING: Removing existing tedana_robustica folder"
   echo "++ ============================================================= ++"
   rm -rf tedana_robustica
fi
pwd 

if [[ "${DATASET}" == "discovery" ]]; then 
   ECHOTIMES="13.9 31.7 49.5" 
else 
   ECHOTIMES="13.7 30.0 47.0" 
fi
echo "++ Dataset Type = ${DATASET} --> Echo Times = ${ECHOTIMES}"

# Run tedana with robustica
tedana -d pb03.${SBJ}.r01.e01.volreg+tlrc.HEAD \
          pb03.${SBJ}.r01.e02.volreg+tlrc.HEAD \
          pb03.${SBJ}.r01.e03.volreg+tlrc.HEAD \
        -e ${ECHOTIMES}                        \
        --mask mask_epi_anat.${SBJ}+tlrc.HEAD  \
        --verbose                              \
        --out-dir tedana_robustica             \
        --convention orig                      \
        --ica-method robustica                 \
        --seed 42                              \
        --n-robust-runs 30


# Scale each echo separately
# --------------------------
echo "++ Correcting space in the header"
echo "================================="
3drefit -space MNI_2009c_asym tedana_robustica/*.nii.gz

echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="
