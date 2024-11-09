"""Data class dealing with mouse data.
author: @gergelyturi
date: 2023-10-16
204-11-09: major refactoring. this used to be the ImagingData class,
now it is the MouseData class"""

from dataclasses import dataclass
from os import walk
from os.path import join
from typing import List


@dataclass
class MouseData:
    """
    Class representing data folders belonging to a mouse.
    
    Attributes
    ----------
    mouse_id : str
        The identifier for the mouse.
    root_folder : str
        The root directory where the mouse data is stored.
    """

    mouse_id: str
    root_folder: str = "/data2/gergely/invivo_DATA/sleep"

    def __post_init__(self):
        """
        Initializes the mouse data object.
        
        Raises
        ------
        ValueError
            If `mouse_id` or `root_folder` is not provided.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.root_folder:
            raise ValueError("root_folder must be specified")
        self.mouse_folders = join(self.root_folder, self.mouse_id)

    def _find_folders(self, condition: callable) -> List[str]:
        """
        General helper function to find folders that meet a certain condition.

        Parameters
        ----------
        condition : callable
            A function that takes a folder name and returns a boolean indicating whether
            the folder meets the desired condition.

        Returns
        -------
        List[str]
            A list of paths to the folders that meet the condition.
        
        Raises
        ------
        ValueError
            If no folders meeting the condition are found.
        """
        folders = []
        for dirpath, dirnames, _ in walk(self.mouse_folders):
            for dirname in dirnames:
                if condition(dirname):
                    folders.append(join(dirpath, dirname))
        if not folders:
            raise ValueError(f"No folders meeting the condition were found in {self.mouse_folders}")
        return folders

    def find_s2p_folders(self) -> List[str]:
        """
        Finds all Suite2p folders for a given mouse.

        Returns
        -------
        List[str]
            A list of all Suite2p folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no Suite2p folders are found.
        """
        return self._find_folders(lambda dirname: dirname.startswith("suite2p"))

    def find_tseries_folders(self) -> List[str]:
        """
        Finds all TSeries folders for a given mouse.

        Returns
        -------
        List[str]
            A list of all TSeries folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no TSeries folders are found.
        """
        return self._find_folders(lambda dirname: dirname.startswith("TSeries") and not dirname.endswith(".sima"))

    def find_sima_folders(self) -> List[str]:
        """
        Finds all .sima folders for a given mouse.

        Returns
        -------
        List[str]
            A list of all .sima folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no .sima folders are found.
        """
        return self._find_folders(lambda dirname: dirname.endswith(".sima"))

    def find_eeg_folders(self) -> List[str]:
        """
        Finds all EEG folders for a given mouse.

        Returns
        -------
        List[str]
            A list of all EEG folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no EEG folders are found.
        """
        return self._find_folders(lambda dirname: dirname == "eeg")

    def find_behavior_folders(self) -> List[str]:
        """
        Finds all behavior folders for a given mouse.

        Returns
        -------
        List[str]
            A list of all behavior folders found in the mouse folder.
        
        Raises
        ------
        ValueError
            If no behavior folders are found.
        """
        return self._find_folders(lambda dirname: dirname == "behavior")
    