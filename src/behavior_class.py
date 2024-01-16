"""Data class dealing with behavior data.
author: @gergelyturi
date: 2023-10-16"""

from dataclasses import dataclass, field
from os.path import join
from os import walk

import numpy as np
import pandas as pd

from mouse_class import Mouse


@dataclass
class behaviorData(Mouse):
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

        print(f"Searching for behavior folders in {self.root_folder}")
        folders = []
        for dirpath, dirnames, subdirnames in walk(self.root_folder):
            if "behavior" in dirnames or "behavior" in subdirnames:
                folders.append(join(dirpath, "behavior"))
        if len(folders) == 0:
            raise ValueError(f"No behavior folders found in {self.root_folder}")
        return folders

    def processed_velocity(
        self, behavior_folder: str = None, file_name="filtered_velo.csv"
    ):
        """Return the processed velocity of the mouse."""
        try:
            filtered_velocity = pd.read_csv(
                join(behavior_folder, file_name), index_col=None
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find processed velocity file in {behavior_folder}, or it named other than 'filtered_velo.csv'"
            )
        return filtered_velocity

    @staticmethod
    def define_immobility(
        velocity: pd.Series,
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
            velocity: pandas Series
                The velocity of the mouse.
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
        window_size = int(framerate * min_duration)
        rolling_max_vel = velocity.rolling(
            window_size, min_periods=min_periods, center=center
        ).max()
        mobile_immobile = (rolling_max_vel > threshold).astype(bool)

        return mobile_immobile
