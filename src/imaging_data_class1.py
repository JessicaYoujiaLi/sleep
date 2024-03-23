from dataclasses import dataclass
from os import walk
from os.path import join


@dataclass
class ImagingData:
    """Initializes imaging data belonging to a mouse."""

    mouse_id: str
    root_folder: str = "rootfolder"

    def __post_init__(self):
        """
        Initializes the imaging data object.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.root_folder:
            raise ValueError("root_folder must be specified")
        self.imaging_folders = join(self.root_folder, self.mouse_id)

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
        print(f"Looking for TSeries folders in {self.imaging_folders}")
        folders = []
        for dirpath, dirnames, _ in walk(self.imaging_folders):
            for dirname in dirnames:
                if dirname.startswith("TSeries") and not dirname.endswith(".sima"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No TSeries found in {self.root_folder}")
        return folders
