"""functions for interacting with data in the
Google Drive
Author: Gergely Turi
"""

from os.path import dirname, join

import gspread
import pandas as pd
from google.auth import default
from google.colab import auth

BASE_DATA_PATH = "/gdrive/Shareddrives/Turi_lab/Data/Sleep/2p/"


def useful_datasets():
    """Return a list of useful datasets."""
    auth.authenticate_user()
    creds, _ = default()
    gc = gspread.authorize(creds)

    worksheet = gc.open("useful_datasets").sheet1

    # get_all_values gives a list of rows.
    rows = worksheet.get_all_values()

    # Convert to a DataFrame and render.
    return pd.DataFrame.from_records(rows[1:], columns=rows[0])
