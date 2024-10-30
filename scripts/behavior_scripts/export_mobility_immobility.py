"""
Script to export mobility and immobility data to JSON file.

"""

import logging
import sys
from argparse import ArgumentParser as AP
from os.path import exists, join

from src.classes import behavior_class as bc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="parsing_log.log",
    filemode="a",
)  # Append to the log file

# Add a StreamHandler to print log messages to the terminal
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)


def main():
    """
    Export mobility and immobility data to JSON files.

    Args:
        mouse_ID (str): The ID of the mouse to analyze.
    """
    parser = AP(description="Export mobility and immobility data to JSON files.")
    parser.add_argument("mouse_ID", help="the ID of the mouse to analyze.")
    parser.add_argument("-o", "--overwrite", help="overwrite existing files", action="store_true")
    args = parser.parse_args()

    # Load the behavior data.
    behavior = bc.behaviorData(args.mouse_ID)

    for behavior_folder in behavior.behavior_folders:
        output_path = join(behavior_folder, "mobility_immobility.json")
        if exists(output_path) and not args.overwrite:
            logger.info(f"Skipping folder {behavior_folder} as file already exists.")
            continue
        try:
            logger.info(f"Processing folder: {behavior_folder}")
            processed_velo = bc.processed_velocity(behavior_folder)
            immobility = bc.define_immobility(velocity=processed_velo)            
            immobility.to_json(output_path, orient="records", indent=4)
            logger.info(
                f"Successfully processed and saved data for folder: {behavior_folder}"
            )
        except FileNotFoundError:
            logger.warning(f"Folder {behavior_folder} not found.")
        except Exception as e:
            logger.error(
                f"Failed to process folder {behavior_folder}. Error: {e}", exc_info=True
            )
            # Continue with the next folder without stopping the script


if __name__ == "__main__":
    main()
