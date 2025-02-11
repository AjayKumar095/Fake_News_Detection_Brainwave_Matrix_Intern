
## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## constant variables
RAW_DATA_PATH = os.path.join(os.getcwd(), "Artifacts", "Dataset", "Fake_News_Detection_Dataset.csv")

