import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle as pkl
import joblib
from utils import constant as const


def load_model():
    
    try:
        with open(const.MODEL_PATH, 'rb') as file:
            model = joblib.load(file)
        return model
    except Exception as e:
        print(f"Error in loading the model: {e}")
        return None

def load_vectorizer():

    try:
        with open(const.VECTORIZER_PATH, 'rb') as file:
            vectorizer = joblib.load(file)
        return vectorizer
    except Exception as e:
        print(f"Error in loading the vectorizer: {e}")
        return None   