"""
Script to export mobility and immobility data to JSON files.
TODO: make it not break when the immobility data is empty.
TODO: make it so that the script can take single experiment folders as input.

"""

from argparse import ArgumentParser as AP
from os.path import join
from src import behavior_class as bc

import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="parsing_log.log",
    filemode="a",
)  # Append to the log file

logger = logging.getLogger(__name__)


def main():
    """
    Export mobility and immobility data to JSON files.

    Args:
        mouse_ID (str): The ID of the mouse to analyze.
    """
    parser = AP(description="Export mobility and immobility data to JSON files.")
    parser.add_argument("mouse_ID", help="the ID of the mouse to analyze.")
    args = parser.parse_args()

    # Load the behavior data.
    behavior = bc.behaviorData(args.mouse_ID)

    for behavior_folder in behavior.behavior_folders:
        try:
            logger.info(f"Processing folder: {behavior_folder}")
            processed_velo = behavior.processed_velocity(behavior_folder)
            immobility = behavior.define_immobility(velocity=processed_velo)
            output_path = join(behavior_folder, "mobility_immobility.json")
            immobility.to_json(output_path, orient="records", indent=4)
            logger.info(
                f"Successfully processed and saved data for folder: {behavior_folder}"
            )
        except Exception as e:
            logger.error(
                f"Failed to process folder {behavior_folder}. Error: {e}", exc_info=True
            )
            # Continue with the next folder without stopping the script


if __name__ == "__main__":
    main()
