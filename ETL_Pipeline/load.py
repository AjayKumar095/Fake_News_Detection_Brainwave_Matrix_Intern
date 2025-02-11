## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ETL_Pipeline.transform import DataTransformer
#from .transform import DataTransformer
from utils.logger import logging


def load_data():
    try:
        logging.info("Loading training and test data")
        X_train_TF, X_test_TF, y_train, y_test = DataTransformer().transform_data()
        return (X_train_TF, X_test_TF, y_train, y_test)
    except Exception as e:
        logging.error(f"Error in loading the data: {e}")
        return None