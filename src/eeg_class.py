"""Data class dealing with EEG data.
author: @gergelyturi
date: 2023-10-16"""

from dataclasses import dataclass, field
from os import walk
from os.path import join

import pandas as pd

from src.imaging_data_class import ImagingData


@dataclass
class eegData(ImagingData):
    """
    A class for loading and processing EEG data.
    """

    eeg_folders: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not self.eeg_folders:
            self.eeg_folders = self.find_eeg_folders()

    def find_eeg_folders(self) -> list:
        """
        Finds all eeg folders in a given root folder.

        Args:
            root_folder (str): The root folder to search for eeg folders.

        Returns:
            list: A list of all eeg folders found in the root folder.

        Raises:
            ValueError: If no eeg folders are found in the root folder.
        """

        print(f"Searching for eeg folders in {self.imaging_folders}")
        folders = []
        for dirpath, dirnames, subdirnames in walk(self.imaging_folders):
            if "eeg" in dirnames or "eeg" in subdirnames:
                folders.append(join(dirpath, "eeg"))
        if len(folders) == 0:
            raise ValueError(f"No eeg folders found in {self.imaging_folders}")
        return folders


def import_scored_eeg(
    eeg_folder: str, eeg_file: str, processed: bool = True
) -> pd.DataFrame:
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
    eeg_data_path = join(eeg_folder, eeg_file)
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


def load_processed_velocity_eeg(file_name: str = "velo_eeg.csv") -> pd.DataFrame:
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
        filtered_velocity = pd.read_csv(file_name, index_col=None)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find processed velocity {file_name}, or it named other than 'velo_eeg.csv'"
        ) from e
    return filtered_velocity


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
