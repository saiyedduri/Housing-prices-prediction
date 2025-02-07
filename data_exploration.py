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

from read_data import *

import sklearn.model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class DataExplorer:
    """
    Process: 
        A class for exploratory data analysis (EDA) to summarize, visualize, and analyze datasets.

    Parameters:

        dataframe : pd.DataFrame
            The pandas DataFrame that will be analyzed.
    Attributes:

        `df : pd.DataFrame
            The original dataset provided during initialization.
        numeric_df : pd.DataFrame
            A subset of the DataFrame containing only numeric columns.
        categorical_df : pd.DataFrame
            A subset of the DataFrame containing only categorical columns.`
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.numeric_df = self.df.select_dtypes(include=["number"])
        self.categorical_df = self.df.select_dtypes(include=["object", "category"])

    def summary_data(self) -> None:
        """
        Function: Displays a brief summary of the dataset, which includes 
                    shapes of features,
                    overview of null values present and datatypes of the features
                    overall statistics of the features.
        
        """
        print("====== TOP 5 Rows ======")
        display(self.df.head())

        print("\n=== Shape of the Data ===")
        display(self.df.shape)

        print("\n=== Overview of Null Values and Data Types ===")
        display(self.df.info())

        print("\n=== Descriptive Statistics ===")
        display(self.df.describe())

    def plot_histograms(self, bins: int = 50, figsize: tuple = (20, 10), save_fig: bool = True) -> None:
        """
        Function: Plots histograms for all numeric features in the dataset.

        Parameters
            bins : int, optional (default=50)
                Number of bins for the histogram.
            figsize : tuple, optional (default=(20, 10))
                The size of the figure.
            save_fig : bool, optional (default=True)
                Whether to save the histogram as an image file.
        """
        plt.figure(figsize=figsize)
        self.df.hist(bins=bins, figsize=figsize)
        plt.suptitle("Histograms of Features", fontname="Times New Roman", fontweight='bold', fontsize=16)
        if save_fig:
            plt.savefig("Histograms_of_features.png")
            print("Histogram of features has been saved as 'Histograms_of_features.png'")
        plt.show()

    def pairplot(self, features: Optional[List[str]] = None, x_vars: Optional[List[str]] = None,
                 y_vars: Optional[List[str]] = None, save_fig: bool = True) -> None:
        """
        Function: Creates pairplots (scatterplot matrix) to visualize relationships between features.

        Parameters
            
            features : list of str, optional
                List of column names to include in the pairplot. If None, all features are included.
            x_vars : list of str, optional
                Variables to be plotted on the x-axis.
            y_vars : list of str, optional
                Variables to be plotted on the y-axis.
            save_fig : bool, optional (default=True)
                Whether to save the pairplot as an image file.
        """
        if features is None:
            features = self.df.columns.tolist()

        g = sns.pairplot(self.df[features], x_vars=x_vars, y_vars=y_vars)
        g.fig.suptitle("Pairplot Among the Features", fontname="Times New Roman", fontweight='bold', fontsize=16)
        g.fig.tight_layout()
        plt.show()

        if save_fig:
            g.fig.savefig("scatter_matrix.png", dpi=300, bbox_inches="tight")
            print("Pairplot saved as 'scatter_matrix.png'.")

    def correlation_matrix(self, save_fig: bool = True) -> None:
        """
        Function: Visualizes the correlation matrix among numeric features using a heatmap.

        Parameters:
            save_fig : bool, optional (default=True)
                Whether to save the correlation matrix as an image file.
        """
        self.corr_mat = self.df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.corr_mat, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f")
        plt.suptitle("Correlation Matrix", fontname="Times New Roman", fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.show()

        if save_fig:
            plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")
            print("Correlation matrix saved as 'correlation_matrix.png'.")

    def missing_values_summary(self) -> None:
        """
        Displays a summary of missing values in the dataset, including:
            count of missing values per feature
            total number of rows
            percentage of missing data
        """
        missing_counts = self.df.isnull().sum()
        total_rows = len(self.df)
        missing_info = pd.DataFrame({
            "missing_count": missing_counts,
            "total_count": total_rows,
            "missing_percentage": (missing_counts / total_rows) * 100
        })

        print("\n====== MISSING VALUES SUMMARY ======")
        display(missing_info[missing_info["missing_count"] > 0])

    def categorical_summary(self) -> None:
        """
        Function: Displays the frequency counts of unique values for each categorical feature.
        Returns:
            None
        """
        print("\n====== Categorical Features Summary ======")
        if hasattr(self, "categorical_df"):
            if self.categorical_df.empty:
                print("There are no categorical features in the DataFrame.")
            else:
                for col in self.categorical_df.columns:
                    print(f"====== Categorical Feature: {col} ======")
                    display(self.categorical_df[col].value_counts())
        else:
            print(f"{self.__class__.__name__} does not have the attribute 'categorical_df'.")

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
