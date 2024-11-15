"""Data class dealing with EEG data.
author: @gergelyturi
date: 2023-10-16
204-11-09: major refactoring"""

from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Union

import pandas as pd


@dataclass
class EegData:
    """
    Class for EEG data.
    Initialized with the path to the EEG folder.
    
    """

    eeg_dir: Union[str, Path]

    def __post_init__(self):
        if isinstance(self.behavior_dir, str):
            self.behavior_dir = Path(self.behavior_dir)

    def _load_csv_file(self, file_path: str) -> pd.DataFrame:
        """
        Helper function to load a CSV file with error handling for missing files.

        Parameters:
        ----------
        file_path : str
            The path to the CSV file to be loaded.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the loaded data.

        Raises:
        ------
        FileNotFoundError
            If the file is not found at the specified path.
        """
        try:
            return pd.read_csv(file_path, index_col=None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the file at '{file_path}'. Please ensure the file path is correct.") from e

    def import_scored_eeg(self, eeg_file: str = "sleep.csv", processed: bool = True) -> pd.DataFrame:
        """
        Imports scored EEG data from a CSV file. The file should be located
        in the specified EEG folder.

        Parameters:
        ----------
        eeg_file : str
            The name of the scored EEG file. Default is 'sleep.csv'.
        processed : bool, optional
            If True, the function will add columns for the different brain states.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the scored EEG data with optional brain state columns.
        """
        scored_eeg_data_path = join(self.eeg_folder, eeg_file)
        eeg_df = self._load_csv_file(scored_eeg_data_path)

        # Check if the DataFrame has only one column
        if eeg_df.shape[1] == 1:
            # Rename the column to "score" and create a "time" column from the index
            eeg_df.columns = ["score"]
            eeg_df["time"] = eeg_df.index
            eeg_df = eeg_df[["time", "score"]]  # Ensure it is a DataFrame
        elif eeg_df.shape[1] == 2:
            # If the DataFrame already has two columns, rename them to "time" and "score"
            eeg_df.columns = ["time", "score"]
        else:
            raise ValueError(f"Unexpected number of columns in {eeg_file}: {eeg_df.shape[1]}")

        # Reorder columns to have "time" first
        eeg_df = eeg_df[["time", "score"]]

        if processed:
            # Process scores into brain state columns
            eeg_df["score"] = eeg_df["score"].astype(int)
            eeg_df["awake"] = (eeg_df["score"] == 0).astype(int)
            eeg_df["NREM"] = (eeg_df["score"] == 1).astype(int)
            eeg_df["REM"] = (eeg_df["score"] == 2).astype(int)
            eeg_df["other"] = (eeg_df["score"] == 3).astype(int)

        return eeg_df


    def load_processed_velocity_eeg(self, file_name: str = "velo_eeg.csv") -> pd.DataFrame:
        """
        Loads the processed velocity data from the specified CSV file.

        Parameters:
        ----------
        file_name : str, optional
            The name of the file containing the processed velocity data.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the processed velocity data.
        """
        processed_velo_eeg_file_path = join(self.eeg_folder, file_name)
        return self._load_csv_file(processed_velo_eeg_file_path)

def brain_state_filter(velo_eeg_df: pd.DataFrame, states: list) -> pd.DataFrame:
    """
    Filters the given DataFrame based on the specified brain states.

    Args:
    - velo_eeg_df: A pandas DataFrame containing EEG data.
    - states: A list of strings representing the brain states to filter for.

    Returns:
    - A pandas DataFrame containing the filtered EEG data.
    """
    conditions = {}
    for state in states:
        if state == "awake_immobile":
            conditions[state] = (
                ~velo_eeg_df["NREM"]
                & ~velo_eeg_df["REM"]
                & ~velo_eeg_df["mobile_immobile"]
                & ~velo_eeg_df["other"]
            )
        elif state == "awake_mobile":
            conditions[state] = (
                ~velo_eeg_df["NREM"]
                & ~velo_eeg_df["REM"]
                & velo_eeg_df["mobile_immobile"]
                & ~velo_eeg_df["other"]
            )
        elif state == "NREM":
            conditions[state] = (
                velo_eeg_df["NREM"]
                & ~velo_eeg_df["REM"]
                & ~velo_eeg_df["other"]
                & ~velo_eeg_df["mobile_immobile"]
            )
        elif state == "REM":
            conditions[state] = (
                ~velo_eeg_df["NREM"]
                & velo_eeg_df["REM"]
                & ~velo_eeg_df["other"]
                & ~velo_eeg_df["mobile_immobile"]
            )
        elif state == "other":
            conditions[state] = velo_eeg_df["other"]
        else:
            print("Unknown state:", state)
    return pd.concat(conditions, axis=1)