# created by: @gergelyturi
from google.colab import auth
import gspread
from google.auth import default
from google.colab import drive

import pandas as pd

class googleDrive:

    def __init__(self):
        self.creds, self._ = default()

    def get_gspread_client(self):
        auth.authenticate_user()
        return gspread.authorize(self.creds)

    def load_spreadsheet_data(self, spreadsheet, sheet):
        """
        Loads data from a specified sheet in a given Google
        Spreadsheet.

        Parameters:
        spreadsheet (str): The name of the Google Spreadsheet.
        sheet (str): The name of the sheet in the Spreadsheet.

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