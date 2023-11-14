import numpy as np
from os import walk
from os.path import join, isdir
from dataclasses import dataclass

COMBINED_DIR_NAME = "combined"
PLANE0_DIR_NAME = "plane0"


class DirectoryNotFoundError(Exception):
    """Exception raised for errors in the directory path."""

    pass


@dataclass
class Suite2p:
    """Class for suite2p data"""

    s2p_folder: str

    def _load_data_from_dir(self, subdir_name, signal_source):
        """
        Helper method to load data from a directory.

        Args:
            s2p_folder (str): The path to the directory containing the data.
            subdir_name (str): The name of the subdirectory containing the data.
            signal_source (str): The name of the signal source file. Either "F" or "Fneu".

        Returns:
            Tuple: A tuple containing the iscells and raw_f arrays, or None if the directory does not exist.
        """

        path = join(self.s2p_folder, subdir_name)
        if not isdir(path):
            raise DirectoryNotFoundError(f"Directory {path} does not exist")

        iscells = np.load(join(path, "iscell.npy"))
        raw_f = np.load(join(path, f"{signal_source}.npy"))
        return iscells, raw_f

    def is_cell_signal(self, signal_source: str = "F") -> np.ndarray:
        """
        Returns the signal of the `is_cell` cells in the Suite2p output directory.

        Args:
            s2p_folder (str): The path to the Suite2p output directory.
            signal_source (str, optional): The source of the signal. Defaults to "F".

        Returns:
            np.ndarray: The true signal of the cells.
        """
        try:
            iscells, raw_f = self._load_data_from_dir(COMBINED_DIR_NAME, signal_source)
        except DirectoryNotFoundError:
            iscells, raw_f = self._load_data_from_dir(PLANE0_DIR_NAME, signal_source)

        signal = np.where(iscells[:, 0])[0]
        return raw_f[signal, :]

    def cells(self):
        """
        Returns the cell signals for the fluorescence data.

        Returns:
        numpy.ndarray: An array of cell signals for the fluorescence data.
        """
        return self.is_cell_signal(signal_source="F")

    def npil(self):
        """
        Computes the neuropil signal for each cell in the Suite2p object.

        Returns:
        -------
        npil : numpy.ndarray
            The neuropil signal for each cell.
        """
        return self.is_cell_signal(signal_source="Fneu")

    def spikes(self):
        """
        Returns the spike signal for each cell in the Suite2p object.

        :return: numpy.ndarray
            Array of shape (num_cells, num_frames) containing the spike signal for each cell.
        """
        return self.is_cell_signal(signal_source="spks")
