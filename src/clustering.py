""" Module with functions for clustering and dimensionality reduction.
Author: Gergely Turi
"""

from os.path import join
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import cosine, pdist, squareform


def interval_length_calculator(data, state_column, state_value):
    """
    Calculate the length, start, and stop indices of intervals for a specific state.

    Args:
        data (pd.DataFrame): The DataFrame containing interval data.
        state_column (str): Column name indicating the state for each interval.
        state_value (int): The value of the state to filter intervals by.

    Returns:
        pd.DataFrame: A DataFrame with columns for the interval number ('n'),
        the start index, the stop index, and the length of each interval of 
        the specified state.
    """
    # Ensure data is not altered outside the function
    data = data.copy()

    # Convert the state column to ensure compatibility
    data[state_column] = data[state_column].astype(int)
    
    # Detect changes to and from the target state
    is_target_state = data[state_column] == state_value
    starts = is_target_state & (~is_target_state.shift(fill_value=False))
    stops = (~is_target_state) & is_target_state.shift(fill_value=False)

    # Prepare for data accumulation
    rows = []

    # Iterate over starts
    for start in data[starts].index:
        # Find the corresponding stop
        stop = data[stops & (data.index > start)].index.min()
        # If there's no corresponding stop, use the last index
        if pd.isna(stop):
            stop = data.index[-1]
        
        length = stop - start +1
        rows.append({'start': start, 'stop': stop, 'length': length})
    
    # Create DataFrame from accumulated rows
    result = pd.DataFrame(rows)
    
    # Add interval numbers
    result.insert(0, 'n', range(1, len(result) + 1))
    
    return result

def process_dfof_intervals(dfof: pd.DataFrame, interval_df: pd.DataFrame):
    """
    Process dF/F data based on specified intervals.

    Args:
        dfof (pd.DataFrame): The dF/F data.
        interval_df (pd.DataFrame): The DataFrame with intervals, containing 'start' and 'stop' columns.

    Returns:
        pd.DataFrame: A DataFrame containing the dF/F data filtered by the specified intervals.
    """
    filtered_dfof = pd.DataFrame()

    # Iterate over each interval
    for _, row in interval_df.iterrows():
        start = row['start']
        stop = row['stop']
        
        # Select the dF/F data for the current interval and append it
        current_interval_data = dfof.iloc[:, start:stop+1]  # Including stop index
        filtered_dfof = pd.concat([filtered_dfof, current_interval_data], axis=1)
    
    return filtered_dfof

def calculate_cosine_distance(data: pd.DataFrame, state: str, save_path: Optional[str] = None,
                              normalize: bool = True) -> pd.DataFrame:
    """
    Calculates the cosine distance matrix for the given data.

    Parameters:
    data (pd.DataFrame): The input data.
    state (str): The brain state of the data.
    save_path (Optional[str]): Path to save the distance matrix. If provided, the matrix is saved to this path.

    Returns:
    pd.DataFrame: The cosine distance matrix.
    """
    X = data.values
    # Calculate pairwise cosine distances and convert to square form
    dist = squareform(pdist(X, metric='cosine'))

    # Normalize the distance matrix
    if normalize:
        min_val, max_val = np.nanmin(dist), np.nanmax(dist)
        dist_data = (dist - min_val) / (max_val - min_val)
    else:
        dist_data = dist

    if save_path:
        pd.DataFrame(dist_data).to_csv(join(save_path, f'ts_dist_{state}.csv'), index=False)

    return pd.DataFrame(dist_data)

def sort_distance_matrix(dist_matrix):
    # Convert the cosine distance matrix to a condensed form since linkage
    # expects a condensed distance matrix
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Perform hierarchical/agglomerative clustering
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # Get the order of rows and columns according to the clustering
    sorted_indices = leaves_list(linkage_matrix)
    
    # Sort the distance matrix according to the clustering result
    sorted_dist_matrix = dist_matrix.iloc[sorted_indices, :].iloc[:, sorted_indices]
    
    return sorted_dist_matrix