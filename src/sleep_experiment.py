"""
Class for sleep experiment data
"""
ROOT_FOLDER = "/data2/gergely/invivo_DATA/sleep"

from dataclasses import dataclass
from os import listdir, makedirs
from os.path import join, exists


@dataclass
class SleepExperiment:
    """Class for sleep experiment data"""

    mouse_id: str
    experiment_date: str
    tseries_folder: str
    root_folder: str = ROOT_FOLDER

    def __post_init__(self):
        """
        Initializes the SleepExperiment object.

        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.experiment_date:
            raise ValueError("Experiment date must be a non-empty string")
        if not self.tseries_folder:
            raise ValueError("TSeries folder must be a non-empty string")

        self.root_folder = join(
            self.root_folder, self.mouse_id, self.experiment_date, self.tseries_folder
        )

    def create_folder_structure(self) -> None:
        """Creates a custom folder structure for the experiment within the .sima folder if it exists,
        otherwise in the tseries_folder."""
        base_path = self.root_folder
        sima_folder = None
        print(f"Base path: {base_path}")
        # Check for .sima folder
        try:
            for folder_name in listdir(base_path):
                if folder_name.endswith(".sima"):
                    sima_folder = folder_name
                    break
        except FileNotFoundError:
            print(f"Error: Base path {base_path} does not exist.")
            return

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
