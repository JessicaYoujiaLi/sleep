"""Data class dealing with EEG data.
author: @gergelyturi
date: 2023-10-16"""

from dataclasses import dataclass
from os.path import join

import pandas as pd


@dataclass
class eegData:
    """
    A class for loading and processing EEG data.

    Attributes:
    -----------
    sima_folder : str
        The path to the folder containing the sima data.
    eeg_folder : str
        The path to the folder containing the EEG data.

    Methods:
    --------
    load_processed_velocity_eeg(file_name: str = "velo_eeg.csv") -> pd.DataFrame:
        Returns the processed velocity of the mouse.
    load_scored_eeg(eeg_file: str = "sleep.csv", processed: bool = True) -> pd.DataFrame:
        Imports scored eeg data from a csv file.
    """

    sima_folder: str = None
    eeg_folder: str = None

    def __post_init__(self):
        self.eeg_folder = join(self.sima_folder, "eeg")

    @property
    def load_processed_velocity_eeg(
        self, file_name: str = "velo_eeg.csv"
    ) -> pd.DataFrame:
        """
        Returns the processed velocity of the mouse.

        Parameters:
        -----------
        file_name : str, optional
            The name of the file containing the processed velocity data.

        Returns:
        --------
        filtered_velocity : pandas DataFrame
            The processed velocity data.
        """
        try:
            filtered_velocity = pd.read_csv(
                join(self.eeg_folder, file_name), index_col=None
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find processed velocity file in {self.eeg_folder}, or it named other than 'velo_eeg.csv'"
            ) from e
        return filtered_velocity

    def load_scored_eeg(
        self, eeg_file: str = "sleep.csv", processed: bool = True
    ) -> pd.DataFrame:
        """
        Imports scored eeg data from a csv file.

        Parameters:
        -----------
        eeg_file : str, optional
            The name of the file containing the scored eeg data.
        processed : bool, optional
            If True, the function will add columns for the different brain states.

        Returns:
        --------
        eeg_df : pandas DataFrame
            The scored eeg data.
        """
        eeg_data_path = join(self.eeg_folder, eeg_file)
        eeg_df = pd.read_csv(eeg_data_path, names=["time", "score"], index_col=False)
        if processed:
            # add columns for the different brain states
            eeg_df["score"] = eeg_df["score"].astype(int)
            eeg_df["awake"] = eeg_df["score"] == 0
            eeg_df["NREM"] = eeg_df["score"] == 1
            eeg_df["REM"] = eeg_df["score"] == 2
            eeg_df["other"] = eeg_df["score"] == 3
        return eeg_df
