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
from data_exploration import *


class Preprocessing():
    """
    Function:
        A class for performing preprocessing tasks on a pandas DataFrame, including:
    handling missing values, encoding categorical variables, splitting data..

    Parameters
        dataframe : pd.DataFrame
            The dataset to be preprocessed.

    Attributes
        df : pd.DataFrame
            The original dataset provided during initialization.
        numeric_df : pd.DataFrame
            A subset of the DataFrame containing only numeric columns.
        categorical_df : pd.DataFrame
            A subset of the DataFrame containing only categorical columns.
    """
    def __init__(self,dataframe:pd.DataFrame)->pd.DataFrame:
        self.df=dataframe
    
    def seperate_numeric_categorical(self)->Union[pd.DataFrame,pd.DataFrame]:
        """
        Separates numeric and categorical features from the dataset.

        Returns
        -------
        tuple
            A tuple containing two DataFrames:
                Numeric DataFrame (`numeric_df`)
                Categorical DataFrame (`categorical_df`)
        """
        self.numeric_df=self.df.select_dtypes(include=["number"])
        self.categorical_df=self.df.select_dtypes(include=["object","category"])
        return self.numeric_df,self.categorical_df
        
    def simple_imputer(self,columns:Union[List[str],str]=None,
                       strategy:str="median",
                       fill_value: Optional[Union[str, int, float]]=None)-> pd.DataFrame:
        """
        Function:
          Missing values in the specified columns using a given strategy.

        Parameters:
        columns : list of str or str, optional
            The columns to impute. If None, all columns are considered.
        strategy : str, default='median'
            The imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        fill_value : str, int, or float, optional
            The value to fill if `strategy='constant'`.

        Returns:
        pd.DataFrame
            The DataFrame with imputed values.
        """

        if columns is None:
            columns=self.df.columns
        elif isinstance(columns,str):
            columns=[columns]
        imputer=SimpleImputer(strategy=strategy,
                              fill_value=fill_value)
        self.imputed_data=imputer.fit_transform(self.df[columns])
        self.df[columns]=self.imputed_data
        return self.df
   
    def split_train_test_data(self,dataframe=None,test_ratio=0.2,random_state=42):
        """
        Function:
            Splits the dataset into training and testing sets.

        Parameters:
            dataframe : pd.DataFrame, optional
                The dataset to split. If None, uses the initialized DataFrame.
            test_ratio : float, default=0.2
                The proportion of the dataset to include in the test split.
            random_state : int, default=42
                Controls the shuffling applied before splitting.

        Returns:
            tuple of (train_data,test_data) of (pd.DataFrame,pd.DataFrame)
        """
        if dataframe is None:
            dataframe=self.df
        np.random.seed(random_state)
        indices=np.random.permutation(len(dataframe))
        random_data=self.df.iloc[indices]
        test_length=int(len(dataframe)*test_ratio)
        test_data=random_data[:test_length]
        train_data=random_data[test_length:]
        return train_data,test_data
    
    def convert_discrete_feature(self,continuous_feature:str,
                                 bins:List[float],
                                 labels:List[str],
                                 show_dirscete_hist: bool=True):
       
        """
        Function:
            Converts continuous features into discrete categories using binning.

        Parameters:
            continuous_feature : str
                The name of the continuous feature to convert.
            bins : list of float
                The bin edges to use for discretization.
            labels : list of str
                The labels for the bins.
            show_discrete_hist : bool, default=True
                Whether to plot a histogram of the discrete categories.

        Returns
            pd.DataFrame
                The DataFrame with the new discrete feature.
        """
        self.df["discrete_cat"]=pd.cut(
                                        self.df[continuous_feature],
                                        bins=bins,
                                        labels=labels)
        if show_dirscete_hist:
            self.df["discrete_cat"].hist()
            plt.title("Histogram for discrete data")
        return self.df
   
    def stratified_split(self, preserve_feature: str,
                         test_ratio: float = 0.2,
                         random_state: int = 42) -> Union[pd.DataFrame, pd.DataFrame]:
        """
        Function:
            Splits the data into training and testing sets while preserving the distribution
        of categories in the specified feature.

        Parameters:

        preserve_feature : str
            The feature whose distribution should be preserved during the split.
        test_ratio : float, default=0.2
            The proportion of the dataset to include in the test split.
        random_state : int, default=42
            Controls the shuffling applied before splitting.

        Returns
            tuple of (stratified_train_data,stratified_test_data) of (pd.DataFrame,pd.DataFrame)
        """
        stratified_train_data,stratified_test_data=pd.DataFrame(),pd.DataFrame()
        
        for each_cat in self.df[preserve_feature].unique():
            cat_data=self.df[self.df[preserve_feature]==each_cat]
            cat_train_data,cat_test_data=self.split_train_test_data(cat_data,test_ratio=test_ratio,random_state=random_state)
            stratified_train_data=pd.concat([stratified_train_data,cat_train_data])
            stratified_test_data=pd.concat([stratified_test_data,cat_test_data])
        return stratified_train_data,stratified_test_data
   
    def encode(self, categorical_features: Union[List[str], str] = None,
               with_strategy: str = "OneHotEncoding") -> pd.DataFrame:
        """
        Function:
            Encodes categorical features using the specified encoding strategy.

        Parameters:

            categorical_features : list of str or str, optional
                The features to encode. If None, all categorical features are considered.
            with_strategy : str, default='OneHotEncoding'
                The encoding strategy ('OneHotEncoding' supported).

        Returns:
            pd.DataFrame
                The DataFrame with encoded features.
        """
        if categorical_features is None:
            categorical_features=self.categorical_df.tolist()
        elif isinstance(categorical_features,str):
            categorical_features=[categorical_features]
        
        if with_strategy=="OneHotEncoding":
            
            cat_encoder=OneHotEncoder()
            hot_encoder_arr=cat_encoder.fit_transform(self.categorical_df[categorical_features].values.reshape(-1,1)).toarray()
            cat_df=pd.DataFrame(hot_encoder_arr,columns=cat_encoder.get_feature_names_out(),
                               index=self.df.index)
            
            # drop the origional categories features from the dataframe
            self.df.drop(columns=categorical_features,inplace=True)
            
            # concatenate the encoded features to the dataframe
            self.df=pd.concat([self.df,cat_df],axis=1)
            
            return self.df
            
    def drop_features(self, drop_feature: Union[str, List[str]],
                      dataframes: Union[List[pd.DataFrame], pd.DataFrame] = None) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Function:
            Drops specified features from one or multiple DataFrames.

        Parameters:
            drop_feature : str or list of str
                The feature(s) to drop from the DataFrame(s).
            dataframes : pd.DataFrame or list of pd.DataFrame, optional
                The DataFrame(s) to process. If None, uses the initialized DataFrame.

        Returns:
            pd.DataFrame or list of pd.DataFrame
                The modified DataFrame(s) with the specified features dropped.
        """
        dropped_dfs=[]
        # If dataframes is not provided, use the instance's dataframes
        if dataframes is None:
            dataframes = self.df

        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]
        
        # dropping the feature (i.e, income category) from the dataframes
        for idx,df in enumerate(dataframes):
            # Drop the feature and append the modified DataFrame to the list
            modified_df = df.drop(drop_feature, axis=1)
            dropped_dfs.append(modified_df)
        
        # Return a single DataFrame if only one was processed
        if len(dropped_dfs) == 1:
            return dropped_dfs[0]
            
        return dropped_dfs
    
    
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
    y_test_act=test_data["median_house_value"]
