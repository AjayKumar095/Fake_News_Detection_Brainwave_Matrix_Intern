
## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## constant variables
RAW_DATA_PATH = os.path.join(os.getcwd(), "Artifacts", "Dataset", "Fake_News_Detection_Dataset.csv")

TRAINING_DATA_PATH = os.path.join(os.getcwd(), "Artifacts", "Dataset", "Training_Dataset.csv")
TEST_DATA_PATH = os.path.join(os.getcwd(), "Artifacts", "Dataset", "Test_Dataset.csv")


MODEL_OBJECTS = os.path.join(os.getcwd(), "Models", "models_objects")
MODEL_REPORTS = os.path.join(os.getcwd(), "Models", "reports")

MODEL_PATH = os.path.join(os.getcwd(), "Models", "models_objects", "best_model.pkl")
VECTORIZER_PATH = os.path.join(os.getcwd(), "Models", "models_objects", "vectorizer.pkl")