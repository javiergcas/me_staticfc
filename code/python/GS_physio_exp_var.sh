set -e

source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate generic_2025a
cd /data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/python

python /data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/python/GS_physio_exp_var.py -g ${GS_PATH} -p ${PHYSIO_PATH} -o ${OUTPUT_PATH}
