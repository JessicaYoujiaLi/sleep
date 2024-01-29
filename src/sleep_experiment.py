"""
Class for sleep experiment data
"""

from dataclasses import dataclass
from os import makedirs
from os.path import join, exists


@dataclass
class SleepExperiment:
    """Class for sleep experiment data"""

    mouse_id: str
    tseries_folder: str = None

    def __post_init__(self):
        """
        Initializes the SleepExperiment object.

        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")

    def create_folder_structure(self) -> None:
        """
        Creates a custom folder structure for the experiment within the .sima folder if it exists.
        """
        # Extract the TSeries folder name from the full path
        tseries_folder_name = self.tseries_folder.rstrip("/").split("/")[-1]

        # Construct the .sima folder path
        sima_folder_path = join(self.tseries_folder, f"{tseries_folder_name}.sima")
        print(f"Creating folder structure in {sima_folder_path}")
        if not exists(sima_folder_path):
            print(f"No .sima folder found in {self.tseries_folder}.")
            return

        directories = ["behavior", "eeg", "plots"]
        for dir_name in directories:
            path = join(sima_folder_path, dir_name)
            if not exists(path):
                makedirs(path)
                print(f"Created directory: {path}")
            else:
                print(f"Directory already exists: {path}")
