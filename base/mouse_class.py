import numpy as np
from os import walk
from os.path import join, isdir
from dataclasses import dataclass, field


@dataclass
class Mouse:
    """Class for mouse data"""

    mouse_id: str
    root_folder: str = None
    tseries_folders: list = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the Suite2p object.

        If root_folder is not specified, sets it to the default path.
        If s2p_folders is not specified, finds all suite2p folders in the root folder.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if self.root_folder is None:
            self.root_folder = f"/data2/gergely/invivo_DATA/sleep/{self.mouse_id}"
        if len(self.tseries_folders) is 0:
            self.tseries_folders = self.find_tseries_folders(self.root_folder)

    @staticmethod
    def find_tseries_folders(root_folder: str) -> list:
        """
        Finds all TSeries folders for a given mouse.

        Args:
            root_folder (str): The root folder to search for suite2p folders.

        Returns:
            list: A list of all suite2p folders found in the root folder.

        Raises:
            ValueError: If no suite2p folders are found in the root folder.
        """
        folders = []
        for dirpath, dirnames, _ in walk(root_folder):
            for dirname in dirnames:
                if dirname.startswith("TSeries"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No TSeries found in {root_folder}")
        return folders
