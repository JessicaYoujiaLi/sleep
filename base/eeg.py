# created by: @gergelyturi
"""EEG-related functions and classes for the project."""

from __future__ import print_function
from pathlib import Path

from sleep.base.core import GoogleDrive
import pandas as pd

def load_eeg_velocity_data(mouse: str, day: str, session:str) -> pd.DataFrame:
  d_path = Path(GoogleDrive.shared_drive_data_path(), 'Sleep', '2p',
       'Analysis', 'data', mouse, day, session, 'velo_eeg.csv')
  df = pd.read_csv(d_path)
  return df