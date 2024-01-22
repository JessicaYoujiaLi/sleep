"""
Class for sleep experiment data
"""
ROOT_FOLDER = "/data2/gergely/invivo_DATA/sleep"

from dataclasses import dataclass
from os import walk, listdir, makedirs
from os.path import join, exists

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

