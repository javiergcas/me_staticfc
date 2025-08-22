# 01/29/2025 - Javier Gonzalez-Castillo
# 06/06/2025 - JGC: Script now computes TSNR
# This script will do the steps beyond afni_proc needed to complete the
# (per-echo) TEDANA and TEDANA+GSR pipelines.
#

export OMP_NUM_THREADS=32
set -e

PRJDIR='/data/SFIMJGC_HCP7T/BCBL2024/'
PRCS_DATA_DIR=`echo ${PRJDIR}/prcs_data`
SBJ_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}`
FMRI_DATA_DIR=`echo ${SBJ_DIR}/D03_Preproc_${SES}_NORDIC-${NORDIC}`
ATLAS_PATH=`echo ${ATLASES_DIR}/${ATLAS_NAME}/${ATLAS_NAME}.nii.gz`

echo "++ Working on Subject ${SBJ}/${SES}... post afni proc"
echo " + PRJDIR=${PRJDIR}"
echo " + PRCS_DATA_DIR=${PRCS_DATA_DIR}"
echo " + SBJ_DIR=${SBJ_DIR}"
echo " + FMRI_DATA_DIR=${FMRI_DATA_DIR}"
echo " + ATLAS_PATH=${ATLAS_PATH}"
echo "=============================================================================="

# Enter destination folder
# ------------------------
echo "++ Entering FMRI_DATA_DIR"
echo "========================="
cd ${FMRI_DATA_DIR}
echo " +  `pwd`"

# Scale each echo separately
# --------------------------
echo "++ Getting individual teadna echoes out of the tedana folder"
echo "============================================================"

3drefit -space MNI tedana_${TEDANA_TYPE}/dn_ts_e1.nii.gz
3drefit -space MNI tedana_${TEDANA_TYPE}/dn_ts_e2.nii.gz
3drefit -space MNI tedana_${TEDANA_TYPE}/dn_ts_e3.nii.gz
3drefit -space MNI tedana_${TEDANA_TYPE}/ts_OC.nii.gz

3dcalc -overwrite -b tedana_${TEDANA_TYPE}/dn_ts_e1.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e01.tedana_${TEDANA_TYPE}_dn
3dcalc -overwrite -b tedana_${TEDANA_TYPE}/dn_ts_e2.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e02.tedana_${TEDANA_TYPE}_dn
3dcalc -overwrite -b tedana_${TEDANA_TYPE}/dn_ts_e3.nii.gz -expr b -datum float -prefix pb06.${SBJ}.r01.e03.tedana_${TEDANA_TYPE}_dn
3dcalc -overwrite -b tedana_${TEDANA_TYPE}/ts_OC.nii.gz    -expr b -datum float -prefix pb06.${SBJ}.r01.tedana_${TEDANA_TYPE}_OC

echo "++ Extract global signal from OC timeseries"
echo "==========================================="
# Global regression with no detrending prior to computing GS
    3dROIstats -quiet \
               -mask ../D03_Preproc_${SES}_NORDIC-off/mask_tedana_at_least_one_echo.nii.gz \
                pb06.${SBJ}.r01.tedana_${TEDANA_TYPE}_OC+tlrc.HEAD | awk '{print $1}' > pb06.${SBJ}.r01.tedana_${TEDANA_TYPE}_OC.GS.1D
    1d_tool.py -overwrite -infile pb06.${SBJ}.r01.tedana_${TEDANA_TYPE}_OC.GS.1D -demean -write pb06.${SBJ}.r01.tedana_${TEDANA_TYPE}_OC.GS.demean.1D 


echo "++ Scaling volreg versions of each echo:"
echo "========================================"
for EC in e01 e02 e03
do
    echo " + scaling [${EC}]"
    3dTstat -overwrite -prefix rm.mean_pb06.${SBJ}.r01.${EC}.tedana_${TEDANA_TYPE}_dn pb06.${SBJ}.r01.${EC}.tedana_${TEDANA_TYPE}_dn+tlrc
done

echo "++ Denoising each echo separately (using MEICA bad components)"
echo "=============================================================="
for EC in e01 e02 e03
do
   echo " + Denoising echo [${EC} | ALL]"
   3dTproject -overwrite                                                                      \
               -polort 0                                                                      \
               -input pb06.${SBJ}.r01.${EC}.tedana_${TEDANA_TYPE}_dn+tlrc                     \
               -ort X.nocensor.xmat.${EC}.1D                                                  \
               -prefix errts.${SBJ}.r01.${EC}.volreg.tproject_ALL_Tedana-${TEDANA_TYPE}       \
               -mask ../D03_Preproc_${SES}_NORDIC-off/mask_tedana_at_least_one_echo.nii.gz
   3dcalc -overwrite -a rm.mean_pb06.${SBJ}.r01.${EC}.tedana_${TEDANA_TYPE}_dn+tlrc -b errts.${SBJ}.r01.${EC}.volreg.tproject_ALL_Tedana-${TEDANA_TYPE}+tlrc -expr 'a+b'     -prefix errts.${SBJ}.r01.${EC}.volreg.tproject_ALL_Tedana-${TEDANA_TYPE}
   3dcalc -overwrite -a rm.mean_pb06.${SBJ}.r01.${EC}.tedana_${TEDANA_TYPE}_dn+tlrc -b errts.${SBJ}.r01.${EC}.volreg.tproject_ALL_Tedana-${TEDANA_TYPE}+tlrc -expr '100*(b-a)/a' -prefix errts.${SBJ}.r01.${EC}.volreg.spc.tproject_ALL_Tedana-${TEDANA_TYPE}
done

echo "++ Computing Full Brain TSNR for Basic and GSasis"
echo "================================================="
for EC in e01 e02 e03
do
  for SCENARIO in ALL_Tedana-${TEDANA_TYPE}
  do
      3dTstat -overwrite -cvarinv -prefix errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}.TSNR.nii errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}+tlrc
      3dcalc  -overwrite \
             -a ../D03_Preproc_${SES}_NORDIC-off/mask_tedana_at_least_one_echo.nii.gz \
             -b errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}.TSNR.nii \
             -expr 'a*b' \
             -prefix errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}.TSNR.nii
      # Compute TSNR at the whole-brain level
      compute_ROI_stats.tcsh                                                           \
         -out_dir    tsnr_stats_regress                                                \
         -stats_file tsnr_stats_regress/TSNR_FB_${EC}_${SCENARIO}.txt                  \
         -dset_ROI   ../D03_Preproc_${SES}_NORDIC-off/mask_epi_anat.${SBJ}+tlrc        \
         -dset_data  errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}.TSNR.nii       \
         -rset_label brain                                                             \
         -rval_list  1

     # Compute TSNR at the ROI level (Same ROIs as afni_proc)
     compute_ROI_stats.tcsh                                                            \
         -out_dir    tsnr_stats_regress                                                \
         -stats_file tsnr_stats_regress/TSNR_ROIs_${EC}_${SCENARIO}.txt                \
         -dset_ROI   ROI_import_MNI_2009c_asym_resam+tlrc                              \
         -dset_data  errts.${SBJ}.r01.${EC}.volreg.tproject_${SCENARIO}.TSNR.nii       \
         -rset_label MNI_2009c_asym                                                    \
         -rval_list  ALL_LT

  done
done

# Extract ROI Timeseries
# ----------------------
echo "++ Extracting ROI Timeseries per echo for atlas [${ATLAS_NAME}]"
echo "==============================================================="
for EC in e01 e02 e03
do
  for INTERP_MODE in ALL 
  do
    for SCENARIO in Tedana-${TEDANA_TYPE}
       do
       echo " + Extracting ROI Timeseries and connectivity for [${EC}]"
       3dNetCorr -overwrite -push_thru_many_zeros                                                              \
                 -mask ../D03_Preproc_${SES}_NORDIC-off/mask_tedana_at_least_one_echo.nii.gz                   \
                 -in_rois ${ATLAS_PATH}                                                                        \
                 -inset  errts.${SBJ}.r01.${EC}.volreg.spc.tproject_${INTERP_MODE}_${SCENARIO}+tlrc          \
                 -prefix errts.${SBJ}.r01.${EC}.volreg.spc.tproject_${INTERP_MODE}_${SCENARIO}.${ATLAS_NAME}

       3dROIstats -quiet \
                  -mask ${ATLAS_PATH} \
                  errts.${SBJ}.r01.${EC}.volreg.spc.tproject_${INTERP_MODE}_${SCENARIO}+tlrc > errts.${SBJ}.r01.${EC}.volreg.spc.tproject_${INTERP_MODE}_${SCENARIO}.${ATLAS_NAME}_000.netts
    done 
  done
done
pwd
echo "++ ========================="
echo "++ Script finished correctly"
echo "++ ========================="
