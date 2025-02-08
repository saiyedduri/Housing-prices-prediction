import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from pandas.plotting import scatter_matrix

import os
import urllib.request
import tarfile
from typing import List,Optional,Any,Union,Dict

import sklearn.model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from read_data import *
from data_exploration import *
from data_preprocessing import *
from model_selection import *


class DataPipeline:
    """
    A pipeline to automate the process of:
        Data retrieval and loading
        Data exploration and
        Preprocessing 
        Model selection & evaluation
    """

    def __init__(self, data_url_path, dataset_folder):
        self.folder_path = dataset_folder
        self.file_path = os.path.join(self.folder_path, "housing.tgz")
        self.url_path = data_url_path
        self.housing_data = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.model_selector = None

    def load_data(self):
        """Downloads and reads the dataset."""
        download_data(url_path=self.url_path, folder_path=self.folder_path, file_path=self.file_path)
        self.housing_data = read_data(folder_path=self.folder_path, file_name="housing.csv")
        print("Data successfully loaded.")

    def explore_data(self):
        """Performs exploratory data analysis."""
        explorer = DataExplorer(self.housing_data)
        explorer.summary_data()
        explorer.plot_histograms()
        explorer.correlation_matrix(save_fig=True)
        explorer.missing_values_summary()
        explorer.categorical_summary()
        explorer.pairplot(features=["median_house_value", "median_income", "total_rooms", "housing_median_age"])
        print("Data exploration complete.")

    def preprocess_data(self):
        """Handles preprocessing steps such as missing values, feature engineering, and transformations."""
        preprocessor = Preprocessing(self.housing_data)

        # Separate numeric & categorical features
        numeric_df, categorical_df = preprocessor.seperate_numeric_categorical()

        # Handle missing values using median imputation
        preprocessor.simple_imputer(columns=numeric_df.columns.tolist(), strategy="median")

        # Create an income category feature
        df_with_income_cat = preprocessor.convert_discrete_feature(
            continuous_feature="median_income",
            bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
        )

        # Stratified train-test split based on income category
        strat_train, strat_test = preprocessor.stratified_split(preserve_feature="discrete_cat", test_ratio=0.2)
        
        # Drop unnecessary features
        train_data, test_data = preprocessor.drop_features(
            drop_feature=["discrete_cat", "ocean_proximity"],
            dataframes=[strat_train, strat_test]
        )

        self.train_data,self.test_data = train_data,test_data
        print("Data preprocessing complete.")

    def prepare_train_test_data(self):
        """Prepares training and testing datasets by setting target variable & scaling features."""
        
        # Create an instance of Preprocessing using the original housing data
        preprocessor = Preprocessing(self.housing_data)
        
        # Define target variable
        self.y_train = np.log1p(self.train_data["median_house_value"])  # Log-transform the target
        self.y_test_act=self.test_data["median_house_value"]

        # Define feature matrices BEFORE splitting train/test data
        self.X_train, self.X_test = preprocessor.drop_features(
            drop_feature="median_house_value",
            dataframes=[self.train_data, self.test_data]  # Using full dataset
        )

        # Standardizing feature matrices
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        
        print("Training and testing data prepared.")

    def select_best_model(self):
        """Performs model selection using cross-validation and evaluates models."""
        self.model_selector = ModelSelection()
        self.model_selector.cross_validation(
            X_train=self.X_train,
            y_train=self.y_train,
            cv=5,
            scoring=["neg_mean_absolute_error", "neg_mean_squared_error", "r2", "neg_mean_absolute_percentage_error"],
            n_jobs=4
        )
        self.model_selector.plot_model_comparisions()
        print("Model selection complete.")

    def predict_targets(self,best_model):
        "Compute the target labels of the test data"
        best_model.fit(self.X_train,self.y_train)
        y_pred_logs=best_model.predict(self.X_test)
        # Converting back to original units from logarithmic values
        self.y_pred=np.expm1(y_pred_logs)
        return self.y_pred
    def compute_accuracy(self):
        "Compute the accuracy of model"
        return r2_score(self.y_test_act,self.y_pred)


    def run_pipeline(self):
        """Runs the full pipeline end-to-end."""
        print("Running Data Pipeline...")
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.prepare_train_test_data()
        self.select_best_model()
        print("Data Pipeline Execution Complete!")


# Running the pipeline
if __name__ == "__main__":
    pipeline = DataPipeline(data_url_path="https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz",
                            dataset_folder="datasets/housing_data")
    pipeline.run_pipeline()
    best_model=RandomForestRegressor()
    y_pred=pipeline.predict_targets(best_model)
    display(pd.DataFrame({"Actual_house_prices($)":pipeline.y_test_act,
              "Predicted_house_prices($)":y_pred
              }))
    accuracy=pipeline.compute_accuracy()
    print(f"The accuracy of the best model selcted is {accuracy*100}%")