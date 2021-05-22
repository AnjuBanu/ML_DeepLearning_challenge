# Importing libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
# Splitting the data into test and train
housing = pd.read_csv("housing.csv")
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.,1.5,3.0,4.5,6.,np.Inf],
                               labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=23)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    train_set_strat=housing.loc[train_index]
    test_set_strat = housing.loc[test_index]
housing_data = train_set_strat.copy()                                               # Creating dataset copy

# Geographical scatter plot of the data
housing_data.plot(kind="scatter", x="longitude", y="latitude")
plt.title("California district geographical plot")
plt.savefig("img1sds")
# Geographical scatter plot with better visualization with high density areas
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)           # Plot based on high density of data points
plt.title("California district geographical plot based on density")
plt.savefig("img2")
# Geographical scatter plot with population and house value visualization
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                  s = housing_data["population"]/100, label = "Population",
                  c = "median_house_value",cmap = plt.get_cmap("jet"), colorbar = True)
plt.title("Population and house value distribution")
plt.legend()
plt.savefig("img3")

# Looking for Correlation between different attributes
corr_matrix = housing_data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing_data[attributes], alpha=0.2, figsize=(12,8))
plt.savefig("corr")
# Plotting the important parameter median house value with median income
housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1 )
plt.savefig("median")

# Experimenting with Attribute combination and cleaning up the data
housing_data["rooms_per_household"] = housing_data["total_rooms"] / housing_data["households"]
housing_data["bedrooms_per_rom"] = housing_data["total_bedrooms"] / housing_data["total_rooms"]
housing_data["population_per_household"] = housing_data["population"] / housing_data["households"]
corr_matrix = housing_data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))