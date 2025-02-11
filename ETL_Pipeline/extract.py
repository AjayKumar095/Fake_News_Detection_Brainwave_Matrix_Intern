## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd 
from utils import constant as const
from transform import DataTransformer
from utils import logger 


def extract_data():
    try:
        logger.info(f"Reading the data from the file: {const.RAW_DATA_PATH}")
        data = pd.read_csv(const.RAW_DATA_PATH)
        transformer = DataTransformer(data)
        training_path, test_path =transformer.transform_data()
        logger.info(f"Data read successfully from the file: {const.RAW_DATA_PATH}")
        return (training_path, test_path)
    except Exception as e:
        logger.error(f"Error in reading the data from the file: {e}")
        return None

