import argparse
from src.imaging_data_class import ImagingData
from src.sleep_experiment import SleepExperiment


def main():
    parser = argparse.ArgumentParser(description="Process the sleep experiment data.")
    parser.add_argument("mouse_id", type=str, help="Enter the mouse ID")
    args = parser.parse_args()

    imaging_data = ImagingData(args.mouse_id)
    tseries_folders = imaging_data.find_tseries_folders()

    for tseries_folder in tseries_folders:
        experiment = SleepExperiment(args.mouse_id, tseries_folder=tseries_folder)
        experiment.create_folder_structure()


if __name__ == "__main__":
    main()
