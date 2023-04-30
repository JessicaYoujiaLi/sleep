# created by: @gergelyturi
"""functions for cross correlation analysis"""
from pathlib import Path
import collections

import numpy as np
import pandas as pd

import scipy.stats as stats

def brain_state_filter(velo_eeg_df: pd.DataFrame, states:list) -> tuple:
    """
    sets up boolean masks for brain states

    Paramaters:
    ===========
    velo_eeg_df: pandas DataFrame
        contains a 'filtered_velo' columns with the velocity and columns
        with brain states. See eeg.import_eeg()
    states: list
        list of the possible brain states: NREM, REM, awake
        
    Return:
    =======
    filters: dictionary
        the keys correspond to boolean values
    states: list
        the states used as an input plus 'locomotion'

    """
    l1 = ['NREM', 'REM', 'awake']
    filters = {}
    if collections.Counter(states) == collections.Counter(l1):
        print(f'Making filters for {l1} and locomotion')
        awake = ((velo_eeg_df['NREM']==False) &
             (velo_eeg_df['REM']==False) &
             (velo_eeg_df['filtered velo']<0.1))
        nrem = ((velo_eeg_df['NREM']) &
                (velo_eeg_df['filtered velo']<0.1))
        rem = ((velo_eeg_df['REM']) &
               (velo_eeg_df['filtered velo']<0.1))
        locomotion = ((velo_eeg_df['NREM']==False) &
              (velo_eeg_df['REM']==False) &
              (velo_eeg_df['filtered velo']>=0.1))
        filters = {'awake':awake,
                  'NREM': nrem,
                  'REM':rem,
                  'locomotion':locomotion}        
    else:
        print('no REM, making filters for NREM, awake and locomotion')
        awake = ((velo_eeg_df['NREM']==False) &
             (velo_eeg_df['REM']==False) &
             (velo_eeg_df['filtered velo']<0.1))
        nrem = ((velo_eeg_df['NREM']) &
                (velo_eeg_df['filtered velo']<0.1))
        locomotion = ((velo_eeg_df['NREM']==False) &
              (velo_eeg_df['REM']==False) &
              (velo_eeg_df['filtered velo']>=0.1))
        filters = {'awake': awake,
                  'nrem': nrem,
                  'locomotion':locomotion}
    states.append('locomotion')
    return filters, states

def upper(df: pd.DataFrame) -> pd.DataFrame:
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate
    matrix from upper triangle.
    
    Source: towardsdatascience.com
        how-to-measure-similarity-between-two-correlation-matrices

    Args:
      df: pandas or numpy correlation matrix

    Returns:
        pd.DataFrame with the values from the upper triangle
    '''
    if isinstance(df, pd.DataFrame):
        df = df.values
    elif not isinstance(df, np.ndarray):
        raise TypeError('Must be pandas or numpy correlation matrix')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]

def data_loader(data_dir: str, data_type: str, mouseID: str,
                 day: str, sessionID: str) -> dict:
    """loads calcium, spike, eeg, cellular data from one session
    Parameters:
    ===========
    data_dir: str
    path like object pointing to the location of the data
    mouseID, day, sessionID: str
    data_type: str
    'dfof' or 'spikes'

    Retrurns:
    =========
    dictionary containing the data
    """
    if data_type != ('dfof' or 'spikes'):
        raise ValueError('data_type must be "dfof" or "spikes"')
    data_loc = Path(data_dir).joinpath(mouseID, day, sessionID)
    data = pd.read_csv(Path(data_loc).joinpath(data_type+'.csv')).set_index('roi_label')

    # loading stat results for significantly up and downregulated cells
    # during NREM
    sig_cells = pd.read_csv(Path(data_loc).joinpath('Significant_paired_DABEST_NREM.csv'))
    print(f'number of cells in this recording: {len(sig_cells)}')

    # loadig EEG and behavior data
    eeg_velocity = pd.read_csv(Path(data_loc).joinpath('velo_eeg.csv'))
    return { data_type: data,
            'significant_cells': sig_cells,
            'eeg_velocity':eeg_velocity}

def significant_cells(cell_data: pd.DataFrame, state: str) -> dict:
    """Returns the list of significant cells for a given state
    
    Parameters:
    ===========
    cell_data: pd.DataFrame
        this is coming from the data_loader function    
    state: str
        brain state, e.g. 'NREM', 'REM', 'awake'

    Return:
    =======
    ind_list: dict
    	dict of significant roi_label indices
    """
    cell_data.drop(labels='Unnamed: 0', inplace=True, axis=1)
    cell_data.set_index('roi_label', inplace=True)
    up_ind_list = list(cell_data[
        (cell_data['Direction']=='Upregulated') &
                             (cell_data['State']=='dfof '+state)].index)
    down_ind_list = list(cell_data[
        (cell_data['Direction']=='Downregulated') &
                             (cell_data['State']=='dfof '+state)].index)
    return {'Upregulated': up_ind_list,
            'Downregulated': down_ind_list}
