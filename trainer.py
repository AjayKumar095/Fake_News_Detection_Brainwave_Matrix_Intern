
## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


""" make sure to add dataset in Artifacts/Dataset folder to start the training process """

from ETL_Pipeline.extract import extract_data
from utils import logger

if __name__=="__main__":
    res=extract_data()
    logger.info(f"Training and Test data saved at {res}")