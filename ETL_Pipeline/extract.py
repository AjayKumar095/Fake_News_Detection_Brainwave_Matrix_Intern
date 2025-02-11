## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd 
from utils import constant as const
from utils.logger import logging 


def extract_data():
    try:
        logging.info(f"Reading the data from the file: {const.RAW_DATA_PATH}")
        data = pd.read_csv(const.RAW_DATA_PATH)
        return data
    except Exception as e:
        logging.error(f"Error in reading the data from the file: {e}")
        return None

