# Import Libraries
import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import sweetviz as sw
import numpy as np
# Setting the path and directories to fetch the real world data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("dataset","housing")
HOUSING_URL =  DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
# Fetching the data
def fetch_housing_data(housing_url=HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    print(tgz_path)
    print(housing_url)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# Extracting the csv file
def load_housing_csv(housing_path = HOUSING_PATH):
    return pd.read_csv(os.path.join(housing_path,"housing.csv"))
# Loading the csv file and plotting an histogram
housing = load_housing_csv()
def plot_housing():
    housing.hist(bins=50,figsize=(20,15))
    plt.savefig("plot_bin")
    plt.show()
# Generating a report to visualize the spread of data across all the features
def housing_report():
    report = sw.analyze(housing)
    report.show_html("report.html")
# Manual split of test and train data
def split_test_train(data, test_ratio):
    indices = np.random.permutation(len(data))
    test_size = int(len(data)*test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


fetch_housing_data()
load_housing_csv()
plot_housing()
housing_report()
train_data, test_data=split_test_train(housing,0.25)
print (len(train_data))
print (len(test_data))