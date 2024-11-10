"""Class dealing with imaging metadata and data"""
import json
from dataclasses import dataclass
from os.path import join
from pathlib import Path
from typing import Union


@dataclass
class Imaging:
    """Class for imaging metadata and data.
    Initialized with a t-series folder path."""

    tseries_dir: Union[str, Path]

    def __post_init__(self):
        if isinstance(self.tseries_dir, str):
            self.tseries_dir = Path(self.tseries_dir)

    def get_imaging_metadata(self):
        """Returns the imaging metadata from the t-series folder."""
        metadata_file = join(self.tseries_dir, "imaging_metadata.json")
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            return metadata
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from {metadata_file}")

    def multiplane_frame_rate(self):
        """Returns the frame rate for multiplane imaging."""
        metadata = self.get_imaging_metadata()
        if metadata.get('sequence_type') == "multi plane":
            try:
                num_images = int(metadata.get("number of images/plane", 0))
                recording_length = float(metadata.get("recording length in seconds", 0))

                if recording_length > 0:
                    fps = num_images / recording_length
                    return float(round(fps, 2))
                else:
                    return "Recording length is zero or invalid, cannot compute frame rate."
            except (ValueError, TypeError) as e:
                return f"Error calculating frame rate: {e}"
        else:
            return "Not a multiplane sequence"

