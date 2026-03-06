#!/bin/bash
set -e

# Activate conda environment
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate tedana_2025a

# Call the compute_pBOLD program

python /data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/python/compute_pBOLD.py \
      -d ${E01_TS_PATH},${E02_TS_PATH},${E03_TS_PATH} \
      -e ${TE_LIST} \
      -m ${METRIC} \
      -o ${OUT_PATH} 

echo "++ Program finished"