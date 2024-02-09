"""
Script to export mobility and immobility data to JSON files.
TODO: make it not break when the immobility data is empty.
TODO: make it so that the script can take single experiment folders as input.

"""

from argparse import ArgumentParser as AP
from os.path import join
from src import behavior_class as bc


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
        print(behavior_folder)
        processed_velo = behavior.processed_velocity(behavior_folder)
        immobility = behavior.define_immobility(velocity=processed_velo)
        immobility.to_json(
            join(behavior_folder, "mobility_immobility.json"),
            orient="records",
            indent=4,
        )


if __name__ == "__main__":
    main()
