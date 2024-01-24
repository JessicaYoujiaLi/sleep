import argparse
from src import sleep_experiment as sle
from src import imaging_data_class as imd

# WIP


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the sleep experiment data.")

    # Add arguments
    parser.add_argument("mouse_id", type=str, help="Enter the mouse ID")
    parser.add_argument(
        "experiment_date", type=str, help="Enter the experiment date (YYYYMMDD)"
    )
    parser.add_argument(
        "tseries_folder", type=str, help="Enter the TSeries folder name"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments to create the SleepExperiment object
    experiment = sle.SleepExperiment(
        args.mouse_id, args.experiment_date, args.tseries_folder
    )

    # Create the folder structure
    experiment.create_folder_structure()


if __name__ == "__main__":
    main()
