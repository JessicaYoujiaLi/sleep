import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass(cutoff, fs, order=5):
    """
    Design a Butterworth lowpass filter.

    Parameters:
    - cutoff (float): The cutoff frequency of the filter.
    - fs (float): The sampling rate of the input signal.
    - order (int, optional): The order of the filter. Default is 5.

    Returns:
    - b (ndarray): The numerator coefficients of the filter.
    - a (ndarray): The denominator coefficients of the filter.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5, pad_length=150, pad_type="edge"):
    """
    Apply a Butterworth lowpass filter to the input data.

    Parameters:
    - data: numpy array
        The input data to be filtered.
    - cutoff: float
        The cutoff frequency of the filter.
    - fs: float
        The sampling frequency of the data.
    - order: int, optional
        The order of the Butterworth filter (default is 5).
    - pad_length: int, optional
        The length of padding to be applied to the data (default is 150).
    - pad_type: str, optional
        The type of padding to be applied (default is "edge"). See numpy.pad for other 
        options.

    Returns:
    - y: numpy array
        The filtered data.

    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    # Padding the data using numpy's pad function
    data_padded = np.pad(data, (pad_length, pad_length), mode=pad_type)
    # Apply filter
    y_padded = filtfilt(b, a, data_padded)
    # Remove padding
    y = y_padded[pad_length:-pad_length]
    return y

def bandpass_filter_with_padding(data, lowcut, highcut, fs, order=3, pad_length=50):
    """
    Apply a bandpass filter to the data with padding to reduce edge artifacts.

    Parameters:
    - data: array-like, the data to be filtered.
    - lowcut: float, the low cutoff frequency.
    - highcut: float, the high cutoff frequency.
    - fs: int or float, the sampling frequency of the data.
    - order: int, the order of the filter.
    - pad_length: int, the number of samples to pad at each end.

    Returns:
    - filtered_data: array-like, the filtered data.
    """
    # Pad data by repeating the first and last values
    first_val, last_val = data.iloc[0], data.iloc[-1]
    pad_front = np.full(pad_length, first_val)
    pad_back = np.full(pad_length, last_val)
    padded_data = np.concatenate([pad_front, data, pad_back])

    # Filter design
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    # Apply filter
    filtered_padded_data = filtfilt(b, a, padded_data)

    # Remove padding
    filtered_data = filtered_padded_data[pad_length:-pad_length]
    return filtered_data


def calculate_power_over_time(
    signal, window_size, step_size, padding_type="zero", padding_size=None
):
    """
    Calculate the power of a pre-filtered signal over time, with optional padding.

    Parameters:
    - signal: array-like, the pre-filtered signal data.
    - window_size: int, the size of the window to calculate power for.
    - step_size: int, the step size for moving the window.
    - padding_type: str, the type of padding ('zero', 'replicate', 'symmetric').
    - padding_size: int or None, the size of the padding. If None, it defaults to half the window size.

    Returns:
    - times: array, the center time of each window.
    - power: array, the power of the signal over time.
    """
    if padding_size is None:
        padding_size = window_size // 2

    # Apply padding based on the selected method
    if padding_type == "zero":
        padded_signal = np.pad(
            signal,
            (padding_size, padding_size),
            mode="constant",
            constant_values=(0, 0),
        )
    elif padding_type == "replicate":
        padded_signal = np.pad(signal, (padding_size, padding_size), mode="edge")
    elif padding_type == "symmetric":
        padded_signal = np.pad(signal, (padding_size, padding_size), mode="symmetric")
    else:
        raise ValueError(
            "Unsupported padding type. Choose 'zero', 'replicate', or 'symmetric'."
        )

    # Update the calculation to use the padded signal
    n = len(padded_signal)
    power = []
    times = []
    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window = padded_signal[start:end]
        power.append(np.mean(window**2))
        # Adjust time calculation for the padding
        times.append((start + end) / 2 - padding_size)

    return np.array(times), np.array(power)

def downsample_boolean_signal(boolean_signal, target_length):
    """
    Downsamples a boolean signal to match a specified target length.

    Parameters:
    - boolean_signal: array-like, the boolean signal to downsample.
    - target_length: int, the desired length of the downsampled signal.

    Returns:
    - downsampled_signal: numpy array, the downsampled boolean signal.
    """
    original_length = len(boolean_signal)
    factor = original_length / target_length
    downsampled_signal = np.zeros(target_length, dtype=bool)

    for i in range(target_length):
        start = int(i * factor)
        end = int((i + 1) * factor)
        # Use the most frequent value in each segment to determine the value of the downsampled signal
        if np.sum(boolean_signal[start:end]) > (end - start) / 2:
            downsampled_signal[i] = True
        else:
            downsampled_signal[i] = False

    return downsampled_signal