import numpy as np
from os import walk
from os.path import join, isdir
from dataclasses import dataclass, field

from mouse_class import Mouse


@dataclass
class Suite2p(Mouse):
    """Class for suite2p data"""

    s2p_folders: list = field(default_factory=list)

    def __post_init__(self):
        """
        Initializes the Suite2p object.

        If root_folder is not specified, sets it to the default path.
        If s2p_folders is not specified, finds all suite2p folders in the root folder.
        """
        super().__post_init__()
        if not self.s2p_folders:
            self.s2p_folders = self.find_suite2p_folders()
        # if len(self.s2p_folders) is 0:
        #     self.s2p_folders = self.find_suite2p_folders(self.root_folder)

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
        folders = []
        for dirpath, dirnames, subdirnames in walk(self.root_folder):
            if "suite2p" in dirnames or "suite2p" in subdirnames:
                folders.append(join(dirpath, "suite2p"))
        if len(folders) == 0:
            raise ValueError(f"No suite2p folders found in {self.root_folder}")
        return folders

    def true_cells(s2p_folder: str) -> np.ndarray:
        """
        Returns the raw fluorescence of the true cells in the specified Suite2p folder.

        Parameters:
        s2p_folder (str): The path to the Suite2p folder.

        Returns:
        np.ndarray: The raw fluorescence of the true cells.
        """
        if isdir(join(s2p_folder, "combined")):
            print("Found combined folder")
            iscells = np.load(join(s2p_folder, "combined", "iscell.npy"))
            raw_f = np.load(join(s2p_folder, "combined", "F.npy"))  # raw fluorescence
        else:
            print("No combined folder found, using plane0")
            iscells = np.load(join(s2p_folder, "plane0", "iscell.npy"))
            raw_f = np.load(join(s2p_folder, "plane0", "F.npy"))  # raw fluorescence
        cells = np.where(iscells[:, 0])[0]
        true_cells = raw_f[cells, :]
        return true_cells

    def true_npil(s2p_folder) -> np.ndarray:
        """
        Returns the neuropil signal for a given Suite2p output folder.

        Args:
            s2p_folder (str): Path to the Suite2p output folder.

        Returns:
            np.ndarray: The true neuropil signal.
        """
        if isdir(join(s2p_folder, "combined")):
            print("Found combined folder")
            iscells = np.load(join(s2p_folder, "combined", "iscell.npy"))
            raw_npil = np.load(
                join(s2p_folder, "combined", "Fneu.npy")
            )  # raw fluorescence
        else:
            print("No combined folder found, using plane0")
            iscells = np.load(join(s2p_folder, "plane0", "iscell.npy"))
            raw_npil = np.load(
                join(s2p_folder, "plane0", "Fneu.npy")
            )  # raw fluorescence
        cells = np.where(iscells[:, 0])[0]
        true_npil = raw_npil[cells, :]
        return true_npil

    @staticmethod
    def cells(s2p_folder):
        """
        Returns the raw fluorescence of the true cells in the specified Suite2p folder.

        Parameters:
        s2p_folder (str): The path to the Suite2p folder.

        Returns:
        np.ndarray: The raw fluorescence of the true cells.
        """
        return true_cells(s2p_folder)

    @staticmethod
    def npil(s2p_folder):
        """
        Returns the neuropil signal for a given Suite2p output folder.

        Args:
            s2p_folder (str): Path to the Suite2p output folder.

        Returns:
            np.ndarray: The true neuropil signal.
        """
        return true_npil(s2p_folder)
