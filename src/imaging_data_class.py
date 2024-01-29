from dataclasses import dataclass, field
from os import walk
from os.path import join

import numpy as np

ROOT_FOLDER = "/data2/gergely/invivo_DATA/sleep"


@dataclass
class ImagingData:
    """Class for imaging data"""

    mouse_id: str
    root_folder: str = None

    def __post_init__(self):
        """
        Initializes the imaging data object.

        If root_folder is not specified, sets it to the default path.
        If s2p_folders is not specified, finds all suite2p folders in the root folder.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if self.root_folder is None:
            self.root_folder = join(ROOT_FOLDER, self.mouse_id)

    def find_tseries_folders(self) -> list:
        """
        Finds all TSeries folders for a given mouse.

        Args:
            root_folder (str): The root folder to search for TSeries folders.

        Returns:
            list: A list of all TSeries folders found in the root folder.

        Raises:
            ValueError: If no TSeries folders are found in the root folder.
        """
        print(f"Looking for TSeries folders in {self.root_folder}")
        folders = []
        for dirpath, dirnames, _ in walk(self.root_folder):
            for dirname in dirnames:
                if dirname.startswith("TSeries") and not dirname.endswith(".sima"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No TSeries found in {self.root_folder}")
        return folders

    def find_suite2p_folders(self) -> list:
        """
        Finds all suite2p folders in a given root folder.

        Args:
            root_folder (str): The root folder to search for suite2p folders.

        Returns:
            list: A list of all suite2p folders found in the root folder.

        Raises:
            ValueError: If no suite2p folders are found in the root folder.
        """

        print(f"Searching for suite2p folders in {self.root_folder}")
        folders = [
            join(dirpath, "suite2p")
            for dirpath, dirnames, _ in walk(self.root_folder)
            if "suite2p" in dirnames
        ]
        if not folders:
            raise ValueError(f"No suite2p folders found in {self.root_folder}")
        return folders
