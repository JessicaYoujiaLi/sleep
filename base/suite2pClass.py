from dataclasses import dataclass, field
from os.path import join, isdir
from os import walk

import numpy as np


@dataclass
class Suite2p:
    """Class for suite2p data"""

    mouse_id: str
    root_folder: str = None
    s2p_folders: list = field(default_factory=list)

    def __post_init__(self):
        if self.root_folder is None:
            self.root_folder = f"/data2/gergely/invivo_DATA/sleep/{self.mouse_id}"
        if len(self.s2p_folders) is 0:
            self.s2p_folders = self.find_suite2p_folders(self.root_folder)

    @staticmethod
    def find_suite2p_folders(root_folder) -> list:
        folders = []
        for dirpath, dirnames, _ in walk(root_folder):
            if "suite2p" in dirnames:
                folders.append(join(dirpath, "suite2p"))
        if len(folders) == 0:
            raise ValueError(f"No suite2p folders found in {root_folder}")
        return folders

    @staticmethod
    def true_cells(s2p_folder):
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

    @staticmethod
    def true_npil(s2p_folder):
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

    def cells(self, s2p_folder):
        return self.true_cells(s2p_folder)

    def npil(self, s2p_folder):
        return self.true_npil(s2p_folder)
