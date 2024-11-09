import json
from dataclasses import dataclass
from os.path import join

import numpy as np
import pandas as pd


@dataclass
class BehaviorData:
    """Class for behavior data.
    Initialized with the path to the behavior folder.
    """

    behavior_folder: str

    def processed_velocity(self, file_name: str = "filtered_velocity.json") -> np.ndarray:
        """
        Return the processed velocity of the mouse.

        Parameters:
        - file_name (str): The name of the velocity file.

        Returns:
        - np.ndarray: The processed velocity as a NumPy array.

        Raises:
        - FileNotFoundError: If the velocity file is not found in the specified folder.
        """
        try:
            with open(join(self.behavior_folder, file_name), "r") as f:
                processed_velocity = np.array(json.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find processed velocity file in {self.behavior_folder}, or it is named differently."
            )
        return processed_velocity

    def define_immobility(
        self,
        velocity: np.ndarray,
        framerate: float = 10,
        threshold: float = 1.0,
        min_duration: float = 1.0,
        min_periods: int = 1,
        center: bool = True,
    ) -> pd.Series:
        """Define time periods of immobility based on a rolling window of velocity.

        A Mouse is considered immobile if velocity has not exceeded the threshold for the
        previous min_duration seconds.

        Default values for min_duration and threshold are taken from:
        Stefanini...Fusi et al. 2018 (https://doi.org/10.1101/292953)

        Args:
            velocity (np.ndarray): The filtered and processed velocity of the mouse.
            framerate (float): The framerate of the velocity data.
            threshold (float): The threshold value for defining immobility.
            min_duration (float): The minimum duration of immobility.
            min_periods (int): The minimum number of periods to consider immobile.
            center (bool): Whether to center the immobile periods.

        Returns:
            pd.Series: A one-dimensional series of booleans, where True signifies mobile
            times and False signifies immobile times.
        """

        velocity_series = pd.Series(velocity).astype(float)
        window_size = int(framerate * min_duration)
        rolling_max_vel = velocity_series.rolling(
            window_size, min_periods=min_periods, center=center
        ).max()
        mobile_immobile = (rolling_max_vel > threshold).astype(bool)

        return mobile_immobile
