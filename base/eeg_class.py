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

    def import_scored_eeg(self, eeg_file: str, processed: bool = True) -> pd.DataFrame:
        # TODO this always need to generate 'awake_immobile' 'awake_mobile' and 'other' columns
        # awake immoble and mobile is based on the define immobility function in behavior class.
        """
        imports scored eeg data from a csv file. The file should be located
        in a subfolder named 'eeg' within the sima folder

        Parameters:
        ===========
        imaging_exp: ImagingExperiment object
        eeg_file: str
            file name of the scored eeg file
        processed: bool, optional
            if True, the function will add columns for the different brain states
        Return:
        ======
        eeg_df: pandas DataFrame

        """
        eeg_data_path = join(self.eeg_folder, eeg_file)
        eeg_df = pd.read_csv(eeg_data_path, names=["time", "score"])
        if processed:
            eeg_df["score"] = eeg_df["score"].astype(int)
            eeg_df["awake"] = eeg_df["score"] == 0
            eeg_df["NREM"] = eeg_df["score"] == 1
            eeg_df["REM"] = eeg_df["score"] == 2
            eeg_df["other"] = eeg_df["score"] == 3
            # replace True/False with 1/0
            eeg_df[["awake", "NREM", "REM", "other"]] = eeg_df[
                ["awake", "NREM", "REM", "other"]
            ].astype(int)
        return eeg_df

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

    def brain_state_filter(
        self, velo_eeg_df: pd.DataFrame, states: list
    ) -> pd.DataFrame:
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
