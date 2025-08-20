set -e

source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate generic_2025a

python /data/SFIMJGC_HCP7T/BCBL2024/me_staticfc/code/python/GS_kappa_and_rho.py -s ${SBJ} -r ${RUN}
