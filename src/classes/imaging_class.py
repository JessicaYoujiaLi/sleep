"""Class dealing with imaging metadata and data"""

import json
from dataclasses import dataclass
from os.path import join


@dataclass
class Imaging:
    """Class for imaging metaadata and data"""

    tseries_dir: str


    def get_imaging_metadata(self):
        """Returns the imaging metadata from the t-series folder"""
        metadata_file = join(self.tseries_dir, "imaging_metadata.json")
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata

