from src.logger import logging
import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            logging.info(f'Object saved successfully at: {file_path}')
    except Exception as e:
        logging.error(f'Error occurred while saving object: {str(e)}')
        raise CustomException(e, sys)
