import datetime
from dataclasses import dataclass
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np

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

    def time_avg_image(self, save_path=None):
        """
        Plot and display the time-averaged image(s) from the ops_array.

        Args:
            save_path (str): Optional. The file path to save the plot.

        Returns:
            str or None: If save_path is provided, returns the path where the plot is saved. Otherwise, returns None.
        """
        ops_path = join(self.s2p_folder, "ops1.npy")
        ops_array = np.load(ops_path, allow_pickle=True)

        # Check the number of elements in ops_array
        num_elements = len(ops_array)

        # Define a consistent plot size for both scenarios
        plot_width_per_image = 5  # Width per image in inches
        plot_height_per_image = 5  # Height per image in inches

        # Single element handling
        if num_elements == 1:
            fig, ax = plt.subplots(
                figsize=(plot_width_per_image, plot_height_per_image)
            )
            image_data = np.flipud(ops_array[0]["meanImg"])
            ax.imshow(
                image_data, cmap="gray"
            )  # cmap='gray' for grayscale, remove if your images are in color
            ax.set_title("Mean Image")
            ax.axis("off")

        # Multiple elements handling
        else:
            # Set up the subplot grid
            cols = int(np.ceil(np.sqrt(num_elements)))
            rows = int(np.ceil(num_elements / cols))

            # Calculate the total plot size
            total_width = plot_width_per_image * cols
            total_height = plot_height_per_image * rows

            # Create subplots
            fig, axes = plt.subplots(rows, cols, figsize=(total_width, total_height))
            axes = axes.flatten()

            # Loop through each item and plot
            for i, ops in enumerate(ops_array):
                image_data = np.flipud(ops["meanImg"])
                axes[i].imshow(image_data, cmap="gray")
                axes[i].set_title(f"Mean Image {i+1}")
                axes[i].axis("off")

            # Turn off any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")
        plt.tight_layout()

        if save_path is not None:
            filename = f"time_avg_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            full_save_path = join(save_path, filename)

            try: 
                plt.savefig(full_save_path)
                plt.close()  # Close the plot to free up memory
            except Exception as e:
                print(f"Error saving plot to {full_save_path}: {e}")
            return save_path
        else:
            plt.show()
            return None
