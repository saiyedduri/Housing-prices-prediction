import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import urllib.request
import tarfile
from typing import List,Optional,Any,Union,Dict


def download_data(url_path: str, folder_path: str, file_path: str) -> None:
    """
    Function: Downloads a compressed data file from a given URL, saves it locally, and extracts its contents.

    Parameters:
    url_path : str
        The URL from which the data file will be downloaded. 
    folder_path : str
        The local directory where the extracted files will be stored. 
        If the folder does not exist, it will be created.

    file_path : str
        The full local path (including the filename) where the downloaded file will be saved.
        Example: "./data/raw_data.tar.gz"

    Returns:
    None
        This function does not return any value. It performs file download, extraction, 
        and prints the names of the extracted files.
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Download the file from the URL
    urllib.request.urlretrieve(url_path, file_path)
    
    # Extract the contents of the compressed file
    with tarfile.open(file_path, "r") as tar:
        for file_name in tar.getnames():
            print(f"Filenames in the zip file are: {file_name}")
        tar.extractall(folder_path,filter="data")


def read_data(folder_path: str, file_name: str) -> pd.DataFrame:
    """
    Function: 
        Reads a CSV file from the specified folder path and returns it as a pandas DataFrame.

    Parameters:
    folder_path : str
        The path to the folder where the CSV file is located.

    file_name : str
        The name of the CSV file to be read.

    Returns:
        DataFrame
            A pandas DataFrame containing the data from the CSV file.
    """

    data = os.path.join(folder_path, file_name)
    read_data = pd.read_csv(data)
    print(f"Done reading the data. Data is stored in {folder_path}")
    return read_data

if __name__=="__main__":
    folder_path=os.path.join("datasets","housing_data")
    file_path=os.path.join(folder_path,"housing.tgz")
    url_path="https://github.com/ageron/handson-ml2/raw/master/datasets/housing/housing.tgz"
    download_data(url_path=url_path,folder_path=folder_path,file_path=file_path)
    housing_data=read_data(folder_path=folder_path,file_name="housing.csv")













