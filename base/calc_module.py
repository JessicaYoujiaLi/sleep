"""
Module for miscallenious calculations and helper functions.
Needs to be orgainzed.
Author: Gergely Turi
email: gt2253@cumc.columbia.edu
"""

import pandas as pd

def label_consecutive_states(data: pd.Series, state: str='NREM') -> pd.Series:

    """    
    Labels consecutive occurrences of a particular state in a Pandas Series.
    
    Parameters:
    -----------
    data : pandas.Series
        The Series containing booleans referring to the state to be labeled.
    state : str, optional (default='NREM')
        The state to label. At least 100 consecutive occurrences of this state
        will be marked with a label.
        
    Returns:
    --------
    pandas.Series
        A Series with a new index indicating consecutive occurrences of the
        specified state.
        
    Example:
    --------
    >>> data = pd.Series([True, True, True, False, True, True, False, True])
    >>> label_consecutive_states(data, 'NREM')
    0       False
    1    NREM1
    2    NREM1
    3       False
    4    NREM2
    5    NREM2
    6       False
    7    NREM3
    dtype: object

    """

    df = pd.Series(False, name=f'{state}_label', index=data.index)

    consecutive_count = 0
    label_count = 1

    for i, val in data.items():
        if val:
            consecutive_count += 1
            if consecutive_count > 100:
                df.loc[i-consecutive_count+1:i] = f'{state}{label_count}'
                label_count += 1
                consecutive_count = 0

        else:
            consecutive_count = 0

    return df