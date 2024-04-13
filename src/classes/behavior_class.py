"""Data class dealing with behavior data.
author: @gergelyturi
date: 2023-10-16"""

import json
from dataclasses import dataclass, field
from os import walk
from os.path import join

import numpy as np
import pandas as pd

from src.classes.imaging_data_class import ImagingData


@dataclass
class behaviorData(ImagingData):
    """Class for behavior data
    Initializing this class with a mouse ID will automatically find
     all behavior folders for that mouse.

     Example:
        behavior = bc.behaviorData("M1")
    """

    behavior_folders: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not self.behavior_folders:
            self.behavior_folders = self.find_behavior_folders()

    def find_behavior_folders(self) -> list:
        """
        Finds all behavior folders in a given root folder.

        Args:
            root_folder (str): The root folder to search for behavior folders.

        Returns:
            list: A list of all behavior folders found in the root folder.

        Raises:
            ValueError: If no behavior folders are found in the root folder.
        """

        print(f"Searching for behavior folders in {self.imaging_folders}")
        folders = []
        for dirpath, dirnames, _ in walk(self.imaging_folders):
            if "behavior" in dirnames:
                folders.append(join(dirpath, "behavior"))
        if not folders:
            raise ValueError(f"No behavior folders found in {self.imaging_folders}")
        return folders


def processed_velocity(
    behavior_folder: str = None, file_name="filtered_velocity.json"
) -> np.ndarray:
    """
    Return the processed velocity of the mouse.

    Parameters:
    - behavior_folder (str): The path to the folder containing the velocity file.
    - file_name (str): The name of the velocity file.

    Returns:
    - np.ndarray: The processed velocity as a NumPy array.

    Raises:
    - FileNotFoundError: If the velocity file is not found in the specified folder.
    """
    try:
        with open(join(behavior_folder, file_name), "r") as f:
            processed_velocity = np.array(json.load(f))

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find processed velocity file in {behavior_folder}, or it named other than 'filtered_velo.csv'"
        )
    return processed_velocity


def define_immobility(
    velocity: np.ndarray,
    framerate: float = 10,
    threshold: float = 1.0,
    min_duration: float = 1.0,
    min_periods: int = 1,
    center: bool = True,
):
    """Define time periods of immobility based on a rolling window of velocity.

    A Mouse is considered immobile if velocity has not exceeded min_vel for the
    previous min_dur seconds.

    Default values for min_dur and min_vel are taken from:
    Stefanini...Fusi et al. 2018 (https://doi.org/10.1101/292953)

    Args:
        velocity: numpy array
            The filtered and processed velocity of the mouse.
        framerate: float
            The framerate of the velocity data.
        threshold: float
            The threshold value for defining immobility.
        min_duration: float
            The minimum duration of immobility.
        min_periods: int
            The minimum number of periods to consider immobile.
        center: bool
            Whether to center the immobile periods.
    Returns:
        mobile_immobile: pandas Series
            A one-dimensional ndarray of booleans, where True signifies mobile
            times and False signifies immobile times.

    """

    velocity_series = pd.Series(velocity).astype(float)
    window_size = int(framerate * min_duration)
    rolling_max_vel = velocity_series.rolling(
        window_size, min_periods=min_periods, center=center
    ).max()
    mobile_immobile = (rolling_max_vel > threshold).astype(bool)

    return mobile_immobile
