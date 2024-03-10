""" Module with functions for clustering and dimensionality reduction.
Author: Gergely Turi
"""

from os.path import join

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def calculate_cosine_distance(
    data: pd.DataFrame, state: str, save_path: None
) -> pd.DataFrame:
    """
    Calculates the cosine distance matrix for the given data.

    Parameters:
    data (pd.DataFrame): The input data.
    state (str): The brain state ofthe data.
    save_path (None): Path to save the distance matrix. If provided, the matrix is saved to this path.

    Returns:
    pd.DataFrame: The cosine distance matrix.
    """
    X = data.values if isinstance(data, pd.DataFrame) else np.array(data)
    n = X.shape[0]
    dist = np.empty((n, n))
    dist.fill(np.nan)

    for i in range(n - 1):
        for j in range(i + 1, n):
            dist[i, j] = 1 - cosine(X[i, :], X[j, :])

    # Fill the lower triangular part of the matrix to make it symmetric
    for i in range(n):
        for j in range(i + 1, n):
            dist[j, i] = dist[i, j]

    min_val = np.nanmin(dist)
    max_val = np.nanmax(dist)
    dist_data = (dist - min_val) / (max_val - min_val)

    if save_path:
        ts_dist_data = pd.DataFrame(dist_data)
        ts_dist_data.to_csv(join(save_path, f"ts_dist_{state}.csv"), index=False)

    return pd.DataFrame(dist_data)
