"""
Class for sleep experiment data
"""
ROOT_FOLDER = "/data2/gergely/invivo_DATA/sleep"

from dataclasses import dataclass, field
from os import walk
from os.path import join, exists
from dataclasses import dataclass, field

@dataclass
class SleepExperiment():
    """Class for sleep experiment data"""

    mouse_id: str
    experiment_date: str
    tseries_folder: str
    root_folder: str = None

    def __post_init__(self):
        """
        Initializes the SleepExperiment object.

        If root_folder is not specified, sets it to the default path.
        If s2p_folders is not specified, finds all suite2p folders in the root folder.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.experiment_date:
            raise ValueError("Experiment date must be a non-empty string")
        if not self.tseries_folder:
            raise ValueError("TSeries folder must be a non-empty string")
        if self.root_folder is None:
            self.root_folder = join(ROOT_FOLDER, self.mouse_id, self.experiment_date)

    def create_exp_folder_structure(self):
        """
        Creates the folder structure for the experiment.

        Returns:
            None
        """
        

        if not exists(join(self.root_folder, self.tseries_folder, )):
            os.makedirs(self.root_folder)

    def find_tseries_folders(self, root_folder: str) -> list:
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

        folders = []
        for dirpath, dirnames, _ in walk(self.root_folder):
            for dirname in dirnames:
                if dirname.startswith("suite2p"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No suite2p folders found in {self.root_folder}")
        return folders

    def find_suite2p_folders(self) -> list:
        """
        Finds all suite2p folders in a given root folder.