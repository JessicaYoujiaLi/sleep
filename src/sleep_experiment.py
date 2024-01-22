"""
Class for sleep experiment data
"""
ROOT_FOLDER = "/data2/gergely/invivo_DATA/sleep"

from dataclasses import dataclass, field
from os import walk, listdir, makedirs
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

    def create_folder_structure(self) -> None:
        """Creates a custom folder structure for the experiment within the .sima folder if it exists,
        otherwise in the tseries_folder."""
        base_path = join(self.root_folder, self.mouse_id, self.experiment_date, self.tseries_folder)
        sima_folder = None

        # Check for .sima folder
        for folder_name in listdir(base_path):
            if folder_name.endswith('.sima'):
                sima_folder = folder_name
                break

        # Define the base path for new directories
        if sima_folder:
            base_path = join(base_path, sima_folder)
            print(f"Using .sima folder: {sima_folder}")
        else:
            print("No .sima folder found. Using TSeries folder.")

        # Create the directories
        directories = ["behavior", "eeg", "plots"]
        for dir_name in directories:
            path = join(base_path, dir_name)
            if not exists(path):
                makedirs(path)
                print(f"Created directory: {path}")
            else:
                print(f"Directory already exists: {path}")


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

