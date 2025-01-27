import pickle
import sys
import os
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            numerical_columns = [
                'Year', 'Kilometer', 'Length', 'Width', 'Height',
                'Seating Capacity', 'Fuel Tank Capacity'
            ]
            categorical_columns = [
                'Make', 'Fuel Type', 'Transmission', 'Location', 'Color',
                'Owner', 'Seller Type', 'Engine', 'Max Power', 'Max Torque',
                'Drivetrain'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info('Transformation pipelines created successfully.')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = "Price"
            input_feature_train_df = train_df.drop(columns=[target_column_name, 'Model'], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name, 'Model'], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing dataframes.')

            # Transform features and convert to dense arrays
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()

            # Handle missing values in target column
            if target_feature_train_df.isnull().any():
                median_price = target_feature_train_df.median()
                target_feature_train_df.fillna(median_price, inplace=True)
                logging.info(f'Filled missing values in target column with median: {median_price}')

            # Create target arrays with explicit 2D shape
            target_feature_train_arr = target_feature_train_df.to_numpy()[:, np.newaxis]
            target_feature_test_arr = target_feature_test_df.to_numpy()[:, np.newaxis]

            # Log detailed information about the arrays
            logging.info(f'Shape of input_feature_train_arr: {input_feature_train_arr.shape} (Dimensions: {input_feature_train_arr.ndim})')
            logging.info(f'Shape of target_feature_train_arr: {target_feature_train_arr.shape} (Dimensions: {target_feature_train_arr.ndim})')
            logging.info(f'First few elements of target_feature_train_arr: {target_feature_train_arr[:5]}')

            # Concatenate using np.concatenate with axis=1
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_arr), axis=1)
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_arr), axis=1)

            logging.info(f'Train array shape after concatenation: {train_arr.shape} (Dimensions: {train_arr.ndim})')
            logging.info(f'Test array shape after concatenation: {test_arr.shape} (Dimensions: {test_arr.ndim})')

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessing object saved successfully.')

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f'Error occurred in data transformation: {str(e)}')
            raise CustomException(e, sys)



