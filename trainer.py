
## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


""" make sure to add dataset in Artifacts/Dataset folder to start the training process """

from ETL_Pipeline.load import load_data
from utils.logger import logging

if __name__=="__main__":
    X_train_TF, X_test_TF, y_train, y_test=load_data()
    logging.info(f"Training and test data loaded successfully: {X_train_TF.shape}, {X_test_TF.shape}, {y_train.shape}, {y_test.shape}")
    