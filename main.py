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

# Splitting the data into train and test data based on the proportionality of income category
strat_train,strat_test=preprocessor.stratified_split(preserve_feature="discrete_cat",test_ratio=0.2,random_state=42)
df_with_income_cat["discrete_cat"].value_counts()/len(df_with_income_cat)

train_data,test_data=preprocessor.drop_features(drop_feature=["discrete_cat","ocean_proximity"],dataframes=[strat_train,strat_test])

# Set median house value as the target feature
y_train=train_data["median_house_value"]

# Set  samples of remaining features as training and test data dropping median house value
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