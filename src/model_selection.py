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


class ModelSelection():
    """
    A class for comparing multiple regression models using cross-validation 
    and visualizing performance metrics such as R², MAE, MSE, and MAPE.

    Parameters
        regressors : dict, optional
            A dictionary containing custom regressors with model names as keys 
            and regressor instances as values. 
            If None, default models (KNN, DT, RF, SVM, MLP, XGB, LGBM) are used.

    Attributes
        regressors : dict
            A dictionary of regression models for evaluation.
        metrics_dict : pd.DataFrame
            A DataFrame containing evaluation metrics for all models after cross-validation.
    """

    def __init__(self, regressors: Optional[Dict[str, object]] = None):
        self.regressors = {
            "KNN": KNeighborsRegressor(),
            "DT": DecisionTreeRegressor(),
            "RF": RandomForestRegressor(),
            "SVM": SVR(),
            "MLP": MLPRegressor(),
            "XGB": XGBRegressor(),
            "LGBM": LGBMRegressor()
        } if regressors is None else regressors

    def cross_validation(self, 
                         X_train: pd.DataFrame, 
                         y_train: Union[pd.Series, np.ndarray], 
                         cv: int,
                         scoring: List[str] = ["neg_mean_absolute_error", 
                                               "neg_mean_squared_error", 
                                               "r2", 
                                               "neg_mean_absolute_percentage_error"],
                         n_jobs: int = 2) -> pd.DataFrame:
        """
        Function:
            Performs cross-validation on the provided regression models and computes evaluation metrics.

        Parameters:
            X_train : pd.DataFrame
                The feature set for training the models.

            y_train : pd.Series or np.ndarray
                The target variable corresponding to the features.

            cv : int
                Number of cross-validation folds.

            scoring : list of str, optional
                List of scoring metrics to evaluate models. Default includes:
                'neg_mean_absolute_error' (MAE)
                'neg_mean_squared_error' (MSE)
                'r2' (R² Score)
                'neg_mean_absolute_percentage_error' (MAPE)

            n_jobs : int, optional (default=2)
                Number of CPU cores to use for parallel processing. 
        Returns:
            pd.DataFrame
                A DataFrame containing average evaluation metrics (R², MAE, MSE, MAPE)
                for each model, sorted by R² score in descending order.
        """
        results_list = []
        for name, model in self.regressors.items():
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            df = pd.DataFrame({
                "regressor": [name],
                "avg_r2": np.abs([np.mean(cv_results["test_r2"])]),
                "avg_MAE": np.abs([np.mean(cv_results["test_neg_mean_absolute_error"])]),
                "avg_MSE": np.abs([np.mean(cv_results["test_neg_mean_squared_error"])]),
                "avg_MAPE": np.abs([np.mean(cv_results["test_neg_mean_absolute_percentage_error"])]),
            })
            results_list.append(df)

        sorted_results = pd.concat(results_list)
        self.metrics_dict = sorted_results.sort_values(by=["avg_r2"], ascending=False, ignore_index=True)
        return self.metrics_dict

    def plot_model_comparisions(self) -> None:
        """
        Function:
        Plots bar charts to compare regression models based on 
        R², MAE, MSE, and MAPE scores from the cross-validation results.

        Returns:                
            Displays a bar plot comparing the performance of each model.
        """
        labels = self.metrics_dict['regressor']
        r2 = self.metrics_dict['avg_r2']
        mae = self.metrics_dict['avg_MAE']
        mse = self.metrics_dict['avg_MSE']
        mape = self.metrics_dict['avg_MAPE']

        x = np.arange(len(labels))  
        width = 0.25                

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting bars for R², MAE, MSE, MAPE
        bars_r2 = ax.bar(x - width, r2, width, label='R²')
        bars_mae = ax.bar(x, mae, width, label='MAE')
        bars_mse = ax.bar(x + width, mse, width, label='MSE')
        bars_mape = ax.bar(x + 2 * width, mape, width, label='MAPE')

        # Annotating the x-axis tick labels, title, and ylabel
        ax.set_ylabel('Scores')
        ax.set_title('Comparison of Regression Models')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    # Reading the data
    folder_path=os.path.join("datasets","housing_data")
    file_path=os.path.join(folder_path,"housing.tgz")
    url_path="https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz"
    download_data(url_path=url_path,folder_path=folder_path,file_path=file_path)
    housing_data=read_data(folder_path=folder_path,file_name="housing.csv")
    
    # Exploring the data
    explorer = DataExplorer(housing_data)
    explorer.summary_data()      
    explorer.plot_histograms()    
    explorer.correlation_matrix(save_fig=True) 
    explorer.corr_mat["median_house_value"].sort_values(ascending=False)
    explorer.missing_values_summary()  
    explorer.categorical_summary()
    explorer.pairplot(features=["median_house_value","median_income","total_rooms","housing_median_age"])

    #Preprocessing the data
    preprocessor=Preprocessing(housing_data)

    #Seperating data into numeric and categorical datatypes
    numeric_df,categorical_df=preprocessor.seperate_numeric_categorical()

    # Impututation the nan values:
    preprocessor.simple_imputer(columns=numeric_df.columns.tolist(),strategy="median",fill_value=None)

    #creating a income category based on the median income, as it is important to have equal proportion in the training data as the collected data
    df_with_income_cat=preprocessor.convert_discrete_feature(continuous_feature="median_income",bins=[0,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])

    # Encoding the categorical variables with one hot encoder
    # encoded_data=preprocessor.encode(categorical_features="ocean_proximity",with_strategy="OneHotEncoding")

    # Splitting the data into train and test data based on the proportionality of income category
    strat_train,strat_test=preprocessor.stratified_split(preserve_feature="discrete_cat",test_ratio=0.2,random_state=42)
    df_with_income_cat["discrete_cat"].value_counts()/len(df_with_income_cat)

    train_data,test_data=preprocessor.drop_features(drop_feature=["discrete_cat","ocean_proximity"],dataframes=[strat_train,strat_test])

    # Set median house value as the target feature
    y_train=train_data["median_house_value"]

    # Set  sampleds of remaining features as training and test data dropping median house value
    X_train,X_test=preprocessor.drop_features(drop_feature="median_house_value",
                    dataframes=[train_data,test_data])
    
    # Standardizing the data
    scaler=StandardScaler()
    X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
    # Applying log transformation to handle outliers in target variable
    y_train = np.log1p(y_train)
    
    # Selection of the model
    model_selector=ModelSelection()
    model_selector.cross_validation(X_train,y_train,cv=5,
                                scoring=["neg_mean_absolute_error",
                                    "neg_mean_squared_error",
                                    "r2",
                                    "neg_mean_absolute_percentage_error"],n_jobs=4)
    model_selector.plot_model_comparisions()

    
