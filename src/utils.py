from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
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
    


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        logging.info("Starting model evaluation...")

        for i, (model_name, model) in enumerate(models.items()):
            logging.info(f"Evaluating model {i+1}/{len(models)}: {model_name}")
            
            para = params[model_name]
            logging.info(f"Performing GridSearchCV for {model_name}")
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            best_params = gs.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params}")
            
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            logging.info(f"Model {model_name} fitted with best parameters")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"Model: {model_name}")
            logging.info(f"Train R2 Score: {train_model_score:.4f}")
            logging.info(f"Test R2 Score: {test_model_score:.4f}")

            report[model_name] = test_model_score

        logging.info("Model evaluation completed")
        logging.info(f"Model performance report: {report}")

        return report

    except Exception as e:
        logging.error(f"Error occurred while evaluating models: {str(e)}")
        raise CustomException(e, sys)

