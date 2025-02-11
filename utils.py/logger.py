
## Add the parent directory to the path so that the logger can be imported from the parent directory
import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Custom logger definition 
import logging
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)


LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)


logging.basicConfig(
    
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)