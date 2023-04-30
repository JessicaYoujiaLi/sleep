# created by: @gergelyturi
from __future__ import print_function
from google.colab import auth
import gspread
from google.auth import default
from google.colab import drive

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from pprint import pprint
import pandas as pd


def mount_drive():
    drive.mount('/gdrive')


def label_consecutive_states(data: pd.Series, state: str='NREM') -> pd.Series:

    """    
    Labels consecutive occurrences of a particular state in a Pandas Series.
    
    Parameters:
    -----------
    data : pandas.Series
        The Series containing booleans referring to the state to be labeled.
    state : str, optional (default='NREM')
        The state to label. At least 100 consecutive occurrences of this state
        will be marked with a label.
        
    Returns:
    --------
    pandas.Series
        A Series with a new index indicating consecutive occurrences of the
        specified state.
        
    Example:
    --------
    >>> data = pd.Series([True, True, True, False, True, True, False, True])
    >>> label_consecutive_states(data, 'NREM')
    0       False
    1    NREM1
    2    NREM1
    3       False
    4    NREM2
    5    NREM2
    6       False
    7    NREM3
    dtype: object

    """

    df = pd.Series(False, name=f'{state}_label', index=data.index)

    consecutive_count = 0
    label_count = 1

    for i, val in data.items():
        if val:
            consecutive_count += 1
            if consecutive_count > 100:
                df.loc[i-consecutive_count+1:i] = f'{state}{label_count}'
                label_count += 1
                consecutive_count = 0

        else:
            consecutive_count = 0

    return df

class GoogleDrive:

    def __init__(self):
        self.creds, self._ = default()

    def get_gspread_client(self):
        auth.authenticate_user()
        return gspread.authorize(self.creds)

    def load_spreadsheet_data(self, spreadsheet: str, sheet: str="Sheet1") -> pd.DataFrame:
        """
        Loads data from a specified sheet in a given Google
        Spreadsheet.

        Parameters:
        spreadsheet : The name of the Google Spreadsheet.
        sheet : The name of the sheet in the Spreadsheet.

        Returns:
        pd.DataFrame: A DataFrame containing the data from the
        specified sheet.
        """
        gc = self.get_gspread_client()
        workbook = gc.open(spreadsheet)
        values = workbook.worksheet(sheet).get_all_values()
        df = pd.DataFrame.from_records(
            values, columns=values[0]).iloc[1:]
        return df


class Mouse:
    """
    A class representing a mouse. Supposed to read and write a google spreadsheet. WIP.
    """

    def __init__(self, name):
        self.name = name

    def add_mouse_to_database(self, path: str) -> int:
        """
        Appends a row of values to the first sheet of a Google Sheets document.

        Args:
            path (str): A list of values to append to the first row of the sheet.

        Returns:
            int: The number of cells that were updated in the sheet.
        """
        credentials, _ = google.auth.default()
        service = build('sheets', 'v4', credentials=credentials)
        body = {
            "range": "A1:C1",
            "majorDimension": "ROWS",
            "values": [[self.name] + path.split()]
        }
        try:
            response = service.spreadsheets().values().append(
                spreadsheetId='1H8wvotuf1hyx-VHeft5ZvuAH6b_C8t4u_taySKdDneg',
                range="A1:C1",
                valueInputOption="RAW",
                body=body
            ).execute()
            return pprint(response)

        except HttpError as error:
            print(f"An error occurred: {error}")
            return 0


pass
