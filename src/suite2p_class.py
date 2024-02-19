import datetime
import warnings
from dataclasses import dataclass
from os.path import exists, isdir, join

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
            subdir_name (str): The name of the subdirectory containing the data.
            signal_source (str): The name of the signal source file. "F" - cell fluorescence data,
                "Fneu" - neuropil data, "spks" - deconvolved spike data.

        Returns:
            Tuple: A tuple containing the iscells, raw_signal arrays and the plane index array
                or None if the directory does not exist.
        """

        path = join(self.s2p_folder, subdir_name)
        if not isdir(path):
            raise DirectoryNotFoundError(f"Directory {path} does not exist")

        iscells = np.load(join(path, "iscell.npy"))
        raw_signal = np.load(join(path, f"{signal_source}.npy"))
        # Load the plane index
        stat_file = np.load(join(path, "stat.npy"))
        values_iplane = [d["iplane"] for d in stat_file if "iplane" in d]
        plane_index = np.array(values_iplane)
        return iscells, raw_signal, plane_index

    def is_cell_signal(self, signal_source: str = "F", plane=None) -> np.ndarray:
        """
        Returns the signal of the `is_cell` cells in the Suite2p output directory.

        Args:
            signal_source (str, optional): The source of the signal. Defaults to "F".
            plane (int, optional): The plane index. If provided, returns the signal of cells in the specified plane.

        Returns:
            np.ndarray: The true signal of the cells.
        """
        try:
            iscells, raw_signal, plane_index = self._load_data_from_dir(
                COMBINED_DIR_NAME, signal_source
            )
        except DirectoryNotFoundError:
            warnings.warn(f"Combined directory not found, falling back to {PLANE0_DIR_NAME}.")
            iscells, raw_signal, plane_index = self._load_data_from_dir(
                PLANE0_DIR_NAME, signal_source
            )
        if plane is not None:
            signal = np.where(iscells[:, 0] & (plane_index == plane))[0]
            return raw_signal[signal, :]
        else:
            signal = np.where(iscells[:, 0])[0]
            return raw_signal[signal, :]

    def get_cells(self, plane=None):
        """
        Returns the cell signals for the fluorescence data.

        Args:
            plane (int, optional): The plane index. Defaults to None.

        Returns:
            numpy.ndarray: An array of cell signals for the fluorescence data.
        """
        return self.is_cell_signal(signal_source="F", plane=plane)

    def get_npil(self, plane=None):
        """
        Computes the neuropil signal for each cell in the Suite2p object.

        Parameters:
        -----------
        plane : int or None, optional
            The plane index for which to compute the neuropil signal. If None, the signal will be computed for all planes.

        Returns:
        -------
        npil : numpy.ndarray
            The neuropil signal for each cell.
        """
        return self.is_cell_signal(signal_source="Fneu", plane=plane)

    def get_spikes(self, plane=None):
        """
        Returns the spike signal for each cell in the Suite2p object.

        Parameters:
        -----------
        plane: int, optional
            The plane number for which to retrieve the spike signal. If not specified, all planes are considered.
        
        Returns:
        -------
        numpy.ndarray: Array of shape (num_cells, num_frames) containing the spike signal for each cell.
        """
        return self.is_cell_signal(signal_source="spks", plane=plane)
    

    def load_avg_image(self):
        """        
        TODO: make it plane specific
        Plot and display the time-averaged image(s) from the ops_array.

        Args:
            save_path (str): Optional. The file path to save the plot.

        Returns:
            str or None: If save_path is provided, returns the path where the plot is saved. Otherwise, returns None.
        """
        ops_path = join(self.s2p_folder, "ops1.npy")        
        if not exists(ops_path):
            print(f"File not found: {ops_path}")
            return None

        ops_array = np.load(ops_path, allow_pickle=True)
        return ops_array

    def plot_time_avg_image(self, ops_array, save_path=None):
        if ops_array is None:
            return None

        num_elements = len(ops_array)

        plot_width_per_image = 5  # Width per image in inches
        plot_height_per_image = 5  # Height per image in inches

        # Single element handling
        if num_elements == 1:
            fig, ax = plt.subplots(
                figsize=(plot_width_per_image, plot_height_per_image)
            )
            # the image is flipped upside down to match the FOV
            image_data = np.flipud(ops_array[0]["meanImg"])
            ax.imshow(
                image_data, cmap="gray"
            )  
            ax.set_title("Mean Image")
            ax.axis("off")

        # Multiple elements handling
        else:            
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
        self._save_or_display_plot(fig, save_path)
        return fig

    def _save_or_display_plot(self, fig, save_path=None):
        if save_path is not None:
            filename = "time_avg_image.png"
            full_save_path = join(save_path, filename)
            if not exists(full_save_path):
                try:
                    fig.savefig(full_save_path)
                    print(f"Saved plot to {full_save_path}")
                except Exception as e:
                    print(f"Error saving plot to {full_save_path}: {e}")
                finally:
                    plt.close(fig)
            else:
                warnings.warn(f"File already exists: {full_save_path}, saving new file.")
                filename = f"time_avg_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                full_save_path = join(save_path, filename)
                try:
                    fig.savefig(full_save_path)
                    print(f"Saved new plot to {full_save_path}")                    
                except Exception as e:
                    print(f"Error saving new plot to {full_save_path}: {e}")
                finally:
                    plt.close(fig)
            
        else:
            plt.show()
            plt.close(fig)
