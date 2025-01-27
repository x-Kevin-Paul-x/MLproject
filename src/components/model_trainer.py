import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor,BaggingRegressor,VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig: 
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model = None

    def initiate_model_trainer(self, train_array, test_array):
        try:    

            logging.info('Training Process has Started!!')
            logging.info('Splitting train and test input and target feature')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],  
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Support Vector Machine": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "Bagging": BaggingRegressor(),
                "Voting": VotingRegressor(estimators=[('lr', LinearRegression()), ('rf', RandomForestRegressor())]),
                "Linear Regression": LinearRegression()
            }  

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': [4, 8, 16, 32, 64]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'splitter': ['best', 'random'],
                    'max_depth': [4, 8, 16, 32, 64]
                },
                "Support Vector Machine": {
                    'C': [0.1, 1, 10],  
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                },  
                "K-Nearest Neighbors": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                "Gradient Boosting": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [4, 32, 64]
                },
                "AdaBoost": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.1, 0.01, 0.001]
                },
                "Extra Trees": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': [4, 8, 16, 32, 64]
                },
                "Bagging": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_features': [3, 5],
                },
                "Voting": {},
                "Linear Regression": {}
                }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")

            logging.info(f'Best found model on both training and testing dataset')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model  
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
        