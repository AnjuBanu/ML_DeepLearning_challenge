import pandas as pd
import numpy as np
import matplotlib.pyplot as  plt
from zlib import crc32

housing = pd.read_csv("housing.csv")

def test_crc32_split(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32
def split_train_test_by_id(data, ratio, id_column):
    ids = data[id_column]
    in_test_train = ids.apply(lambda id_ : test_crc32_split(id_,ratio))
    return data.loc[~in_test_train],data.loc[in_test_train]

# Generating the test and train data by creating a new index feature
# To be used in scenarios when there is no Identifier column
housing_with_id = housing.reset_index()
test_data_index, train_data_index = split_train_test_by_id(housing_with_id,0.2,"index")

# Generating the test and train data by using existing features
# To be used in scenarios when there is identifier column
housing["id"] = housing["longitude"]*1000 +  housing["latitude"]
test_data_id, train_data_id = split_train_test_by_id(housing,0.2,"id")

# Generating the test and train data by using scikit library
# For Balanced dataset
from sklearn.model_selection import train_test_split
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.,1.5,3.0,4.5,6.,np.Inf],
                               labels=[1,2,3,4,5])
train_set_random,test_set_random = train_test_split(housing,test_size=0.2,random_state=23)

# For Imbalanced dataset
from sklearn.model_selection import StratifiedShuffleSplit
housing["income_cat"].hist()
plt.savefig("income_cat")
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=23)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    train_set_strat=housing.loc[train_index]
    test_set_strat = housing.loc[test_index]

# Comparison of different train-test data split methods
strat_values=(test_set_strat["income_cat"].value_counts() / len(test_set_strat["income_cat"])).sort_index ()
normal_values=(housing["income_cat"].value_counts() / len(housing["income_cat"])).sort_index ()
random_values=(test_set_random["income_cat"].value_counts() / len(test_set_random["income_cat"])).sort_index ()
strat_error= ((strat_values - normal_values) / normal_values)*100
random_error= ((random_values - normal_values) / normal_values)*100
sampling=pd.DataFrame({"Overall":normal_values,
                       "Random":random_values,
                       "Stratified":strat_values,
                       "Random Error %":random_error,
                       "Strat Error %":strat_error,})
