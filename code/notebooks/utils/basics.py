import os.path as osp
import pandas as pd
import numpy as np

PRJ_DIR       = '/data/SFIMJGC_HCP7T/BCBL2024/'
ATLAS_NAME    = 'Schaefer2018_400Parcels_17Networks'

PRCS_DATA_DIR = osp.join(PRJ_DIR,'prcs_data')
ATLASES_DIR   = osp.join(PRJ_DIR,'atlases')
CODE_DIR      = osp.join(PRJ_DIR,'me_staticfc','code')

# Echo Time information for the Spreng Dataset
TES_MSEC_PER_SCANNER = {'1':{'e01':13.7,'e02':30,'e03':47},
                        '2':{'e01':14,'e02':29.96,'e03':45.92}}

# We are only working with data from Site 1
TES_MSEC = TES_MSEC_PER_SCANNER['1']

Power264_cmap = {''}

def read_gen_ss_review_table(file_path):
    """
    Reads the output of AFNI command gen_ss_review_table and
    organizes the contents into a pandas dataframe with meaningful
    column names
    
    Input
    -----
    file_path: path to input file
    
    Returns
    -------
    df: dataframe with the information in the file
    """
    if not osp.exists(file_path):
        print('++ERROR [read_gen_ss_review_table]: input file does not exists')
        return None
    df = pd.read_csv(file_path, sep='\t', header=None)
    original_columns  = [(a,b) for a,b in df.loc[[0,1]].values.T]
    df.drop([0,1], inplace=True)
    new_columns = []
    for (a,b) in original_columns:
        if a=='infile':
            new_columns = new_columns + ['infile']
            continue
        if (a=='echo times'):
            new_columns = new_columns + ['e01','e02','e03']
            continue
        if (a is np.nan):
            continue
        if (a=='orig voxel counts'):
            new_columns = new_columns + ['Nx','Ny','Nz']
            continue
        if (a=='orig voxel resolution'):
            new_columns = new_columns + ['orig Dx','orig Dy','orig Dz']
            continue
        if (a=='final voxel resolution'):
            new_columns = new_columns + ['final Dx','final Dy','final Dz']
            continue
        if (a=='orig volume center'):
            new_columns = new_columns + ['orig volume center x','orig volume center y','orig volume center z']
            continue
        new_columns = new_columns + [a]
    df.columns = new_columns
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except:
            df[c] = df[c]
    print("++ INFO [read_gen_ss_review_table]: Number of scans = %d | Number of metrics per scan = %d" % (df.shape))
    return df
