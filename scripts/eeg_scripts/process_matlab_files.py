"""This script takes YP's matlab files that contain scored data and exports the state data to a csv file.
There are other data in the matlab files that are not exported here.
Author: Gergely Turi
date: 11/10/2023
"""

import argparse as ap
import os
from os.path import dirname, isfile, join

import h5py
import numpy as np


def main(sima_folder: str, mat_file: str, source_eeg_folder: str):
    try:
        # Finding the data
        eeg_folder = join(sima_folder, "eeg", source_eeg_folder)
        mat_path = join(eeg_folder, mat_file)

        # Check if the .mat file exists
        if not isfile(mat_path):
            raise FileNotFoundError(f"MAT file not found at: {mat_path}")

        # Save path
        parent_folder = dirname(eeg_folder)
        csv_path = join(parent_folder, "sleep.csv")

        # Extracting the data
        with h5py.File(mat_path, "r") as f:
            # Check if dataset exists in the file
            if "sleepData" not in f or "state" not in f["sleepData"]:
                raise KeyError("The dataset 'sleepData/state' does not exist in the specified .mat file.")

            # Accessing the dataset
            sleep_data = f["sleepData"]["state"][:]
            sleep_data = sleep_data.reshape(-1)

        # Saving the data to CSV
        np.savetxt(csv_path, sleep_data, delimiter=",", fmt="%d")
        print(f"Data saved to {csv_path}")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")

    except KeyError as key_error:
        print(f"Error: {key_error}")

    except OSError as os_error:
        print(f"OS Error while accessing files: {os_error}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Parsing command-line arguments
    arg_parser = ap.ArgumentParser(description="Extract scored data from YP's matlab files.")
    arg_parser.add_argument("sima_folder", type=str, help="The path to the sima folder.")
    arg_parser.add_argument("mat_file", type=str, help="The name of the matlab file.")
    arg_parser.add_argument("source_eeg_folder", type=str, help="The path to the source EEG folder. E.g. 2024-08-08_17-27-25")
    args = arg_parser.parse_args()

    # Calling main with parsed arguments
    main(args.sima_folder, args.mat_file, args.source_eeg_folder)
