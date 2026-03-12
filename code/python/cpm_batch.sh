set -e

source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate generic_2025a


python /data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/python/cpm_batch.py \
       -b ${BEHAV_FILE} \
       -t ${BEHAV} \
       -l ${OUT_LABEL} \
       -o ${OUT_DIR} \
       -i ${ITER} \
       -f ${FC_PATH} \
       -p ${P_THR} \
       -c ${COR_MODE} \
       -M ${CONFOUND_FILE} \
       ${OTHER_PARS}
