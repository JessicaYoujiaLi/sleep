"""Data class dealing with behavior data.
author: @gergelyturi
date: 2023-10-16"""

from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd


@dataclass
class behaviorData:
    sima_folder: str = None
    behavior_folder: str = None

    def __post_init__(self):
        self.behavior_folder = join(self.sima_folder, "behavior")

    @property
    def processed_velocity(self, file_name="filtered_velo.csv"):
        """Return the processed velocity of the mouse."""
        try:
            filtered_velocity = pd.read_csv(
                join(self.behavior_folder, file_name), index_col=None
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find processed velocity file in {self.behavior_folder}, or it named other than 'filtered_velo.csv'"
            )
        return filtered_velocity

    def define_immobility(
        self,
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
                A one-dimensional ndarray of 0's and 1's, where 1 signifies mobile
                times and 0 signifies immobile times.

        """
        window_size = int(framerate * min_duration)
        rolling_max_vel = velocity.rolling(
            window_size, min_periods=min_periods, center=center
        ).max()
        mobile_immobile = (rolling_max_vel > threshold).astype(int)

        return mobile_immobile
