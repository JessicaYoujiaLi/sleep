import numpy as np
import pandas as pd


class inputOutput:
    """
    A class for handling input/output operations related to neuropil analysis.
    """

    @staticmethod
    def save_results_h5(data: pd.DataFrame, path_name: str, name: str):
        """
        Save the results in an HDF5 file format.

        Args:
            data (pd.DataFrame): The data to be saved.
            path_name (str): The path name where the HDF5 file will be saved.
            name (str): The name of the HDF5 file.

        Returns:
            None
        """
        with pd.HDFStore(f"{path_name}.h5") as store:
            for col in data.columns:
                for idx, cell in enumerate(data[col]):
                    if isinstance(cell, dict) and all(
                        isinstance(cell[key], np.ndarray)
                        for key in ["x_axis", "psd", "mean_ca"]
                    ):
                        # Create a path in HDF5 to save the arrays
                        path_freq = f"{col}/{idx}/x_axis"
                        path_psd = f"{col}/{idx}/psd"
                        path_mean_ca = f"{col}/{idx}/mean_ca"

                        # Save the numpy arrays to the respective paths
                        store.put(path_freq, pd.Series(cell["x_axis"]))
                        store.put(path_psd, pd.Series(cell["psd"]))
                        store.put(path_mean_ca, pd.Series(cell["mean_ca"]))

                        # Replace the cell content with the reference to the paths
                        data.at[idx, col] = {
                            "x_axis": path_freq,
                            "psd": path_psd,
                            "mean_ca": path_mean_ca,
                        }
        data.to_hdf(name, key="main_df", mode="a")

    @staticmethod
    def load_results(path_name: str):
        """
        Load results from an HDFStore file.

        Parameters:
        path_name (str): The path to the HDFStore file.

        Returns:
        pandas.DataFrame: The loaded results DataFrame.
        """
        with pd.HDFStore(path_name) as store:
            loaded_results_df = store["main_df"]
        return loaded_results_df
