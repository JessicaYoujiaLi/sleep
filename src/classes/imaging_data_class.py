from dataclasses import dataclass
from os import walk
from os.path import join


@dataclass
class ImagingData:
    """Initializes imaging data belonging to a mouse."""

    mouse_id: str
    root_folder: str = "/data2/gergely/invivo_DATA/sleep"

    def __post_init__(self):
        """
        Initializes the imaging data object.
        """
        if not self.mouse_id:
            raise ValueError("Mouse ID must be a non-empty string")
        if not self.root_folder:
            raise ValueError("root_folder must be specified")
        self.imaging_folders = join(self.root_folder, self.mouse_id)

    def find_s2p_folders(self) -> list:
        """
        Finds all suite2p folders for a given mouse.

        Args:
            root_folder (str): The root folder to search for suite2p folders.

        Returns:
            list: A list of all suite2p folders found in the root folder.

        Raises:
            ValueError: If no suite2p folders are found in the root folder.
        """
        print(f"Looking for Suite2p folders in {self.imaging_folders}")
        folders = []
        for dirpath, dirnames, _ in walk(self.imaging_folders):
            for dirname in dirnames:
                if dirname.startswith("suite2p"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No Suite2p folders were found in {self.root_folder}")
        return folders

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
    
    def find_sima_folders(self) -> list:
        """
        Finds all .sima folders for a given mouse.

        Args:
            root_folder (str): The root folder to search for sima folders.

        Returns:
            list: A list of all sima folders found in the root folder.

        Raises:
            ValueError: If no sima folders are found in the root folder.
        """
        print(f"Looking for sima folders in {self.imaging_folders}")
        folders = []
        for dirpath, dirnames, _ in walk(self.imaging_folders):
            for dirname in dirnames:
                if dirname.endswith(".sima"):
                    folders.append(join(dirpath, dirname))
        if len(folders) == 0:
            raise ValueError(f"No .sima folders were found in {self.root_folder}")
        return folders