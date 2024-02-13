"""
Methods for calculating the frequency and power spectral density (PSD) of a signal.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def freq_calc(data: pd.Series, fs: int = 10, resolution: float = 0.01):
    """
    Calculate the frequency spectrum of the given data using Welch's method.

    Parameters:
    - data: pd.Series
        The input data for frequency analysis.
    - fs: int, optional
        The sampling frequency of the data. Default is 10.
    - resolution: float, optional
        The desired frequency resolution. Default is 0.01.

    Returns:
    - frequencies: array-like
        The frequencies corresponding to the power spectral density.
    - psd: array-like
        The power spectral density of the input data.
    """
    nperseg = int(fs / resolution)
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, detrend="linear")
    return frequencies, psd


def calculate_autocorrelations(df):
    """
    Calculate autocorrelations for each row in the DataFrame.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing the signals.

    Returns:
    - np.array
        An array of autocorrelations for each row in the DataFrame.
    """
    autocorrelations = []  # To store autocorrelation results
    for index, row in df.iterrows():
        signal = row.values
        autocorr = np.correlate(signal, signal, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]  # Keep second half
        autocorr /= autocorr[0]  # Normalize
        autocorrelations.append(autocorr[:500])
    return np.array(autocorrelations)


def calculate_psd(data, sampling_rate, freq_range=None):
    """
    Calculate the Power Spectral Density (PSD) for a given signal.

    Parameters:
    - data: 2D numpy array of the input signals (signals x time points).
    - sampling_rate: Sampling rate of the data in Hz.
    - freq_range: Optional tuple specifying the frequency range (min_freq, max_freq) for the PSD calculation.

    Returns:
    - freqs: 1D numpy array of frequencies corresponding to the PSD values.
    - psd_values: 2D numpy array of PSD values (signals x PSD values).
    """
    n = data.shape[1]  # Number of data points
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    psd_values = []

    for signal in data:
        # Compute the FFT and PSD
        fft_result = np.fft.rfft(signal)
        psd = np.abs(fft_result) ** 2 / n
        psd_values.append(psd)

    psd_values = np.array(psd_values)

    if freq_range is not None:
        # Limit PSD and frequencies to the specified range
        min_freq, max_freq = freq_range
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs = freqs[freq_mask]
        psd_values = psd_values[:, freq_mask]

    return freqs, psd_values


def plot_psd(
    freqs,
    psd_values,
    ax,
    title="PSD of Autocorrelation",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density",
    label=None,
):
    """
    Plot a line plot of the Power Spectral Density (PSD) values.

    Parameters:
    - freqs: 1D numpy array of frequencies corresponding to the PSD values.
    - psd_values: 2D numpy array of PSD values for each signal (rows for signals, columns for PSD values).
    - ax: matplotlib.axes.Axes object where the plot will be drawn.
    - title: String, title of the plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    """
    # Plot the average PSD across all signals for simplicity
    # You can modify this to plot individual signals or a subset
    avg_psd = np.mean(psd_values, axis=0)
    ax.plot(freqs, avg_psd, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def spectral_density_plot(
    data: pd.DataFrame,
    states: list,
    ax: plt.Axes = None,
    labels: dict = None,
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
    ax.set_xlim(0.01, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
