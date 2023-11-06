"""functions used for neuropil analysis
author: @gergelyturi"""


from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import statsmodels.stats.multitest as smm


class inputOutput:
    pass

    @staticmethod
    def save_results(data: pd.DataFrame, path_name: str, name: str):
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
        with pd.HDFStore(path_name) as store:
            loaded_results_df = store["main_df"]
        return loaded_results_df


def freq_calc(data: pd.Series, fs: int = 10, resolution: float = 0.01):
    """data: npil_eeg data"""
    nperseg = int(fs / resolution)
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, detrend="linear")
    return frequencies, psd


def calc_mean_sem_ci(data: pd.DataFrame, state: str, confidence: float = 0.95):
    """Calculate the mean and standard error of the mean (SEM) for each cell"""
    mean = data.loc[state].mean()
    sem = stats.sem(data.loc[state], axis=0, nan_policy="omit")
    ci = stats.t.ppf((1 + confidence) / 2.0, len(data.loc[state]) - 1)
    return mean, sem, ci


def spectral_density_plot(
    data: pd.DataFrame,
    states: list,
    ax: plt.Axes = None,
    labels: dict = None,
    **kwargs,
) -> plt.Axes:
    """Plot the spectral density of two states
    data: pd.DataFrame
    states: list of two states
    ax: plt.Axes
    labels: dict of labels
        implemented keys: "suptitle", "title"
    save: bool

    returns: plt.Axes
    """
    if labels is None:
        labels = {}

    if ax is None:
        fig, ax = plt.subplots()
        figure_created = True
    else:
        figure_created = False

    if figure_created:
        fig.suptitle(labels.get("suptitle", "Power Spectral Density: Awake vs NREM"))

    # Calculate the standard error of the mean (SEM) for confidence intervals
    state1_mean, state1_sem, ci = calc_mean_sem_ci(data, states[0])
    state2_mean, state2_sem, _ = calc_mean_sem_ci(data, states[1])

    state1_lower = state1_mean - ci * state1_sem
    state1_upper = state1_mean + ci * state1_sem

    state2_lower = state2_mean - ci * state2_sem
    state2_upper = state2_mean + ci * state2_sem

    # Plot the means with semilogy
    x_axis = np.linspace(0, 5, num=len(state1_mean))
    ax.semilogy(x_axis, state1_mean, label=states[0])
    ax.semilogy(x_axis, state2_mean, label=states[1])
    # Plot the confidence intervals with fill_between
    ax.fill_between(x_axis, state1_lower, state1_upper, color="blue", alpha=0.2)
    ax.fill_between(x_axis, state2_lower, state2_upper, color="orange", alpha=0.2)

    ax.legend(loc="lower left")
    ax.set_title(labels.get("title", "Cell"))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    ax.set_xlim(0.01, 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


class statistics:
    pass

    @staticmethod
    def significance_calc(nrem_array: np.array, awake_array: np.array) -> np.array:
        p_values = []

        for nrem_a, awake_a in zip(nrem_array, awake_array):
            stat, p = stats.mannwhitneyu(
                nrem_a, awake_a, alternative="greater"
            )  # alternative='greater' if hypothesis is that NREM values are higher
            p_values.append(p)

        # Multiple testing correction using statsmodels
        reject_list, pvals_corrected, alphacSidak, alphacBonf = smm.multipletests(
            p_values, alpha=0.05, method="bonferroni"
        )
        # reject_list is a boolean array where True means that we reject the null hypothesis for that test
        # pvals_corrected are the adjusted p-values

        # Find indices of significant tests
        significant_cells = np.where(reject_list)[0]
        return significant_cells
