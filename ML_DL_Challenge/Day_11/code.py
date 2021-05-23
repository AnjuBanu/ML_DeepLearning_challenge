# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
# Splitting the data into test and train
housing = pd.read_csv("housing.csv")
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6.,np.Inf],labels=[1,2,3,4,5])
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=23)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    train_set_strat=housing.loc[train_index]
    test_set_strat = housing.loc[test_index]
housing = train_set_strat.drop("median_house_value", axis=1)                                  # Independent Variables
housing_label = train_set_strat["median_house_value"].copy()                                  # Dependent Variables

# Performing Data Cleaning Manually
housing_1 = housing.dropna(subset=["total_bedrooms"])                                         # Option 1: Drop NA row
housing_2 = housing.drop("total_bedrooms", axis=1)                                            # Option 2: Drop complete column
housing_3 = housing["total_bedrooms"].fillna(housing["total_bedrooms"].median(), inplace=True)# Option 3: Replace NA with median values

# Performing Data Cleaning using sklearn for numerical attributes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")                                                    # Using Impute Estimator
housing_num = housing.drop("ocean_proximity", axis=1)                                         # Dropping categorical attributes before imputation
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)              # converting np array back to data frame

# Performing Data Cleaning using sklearn for text or ordinal attributes
from sklearn.preprocessing import OrdinalEncoder
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
ordinal_encoder = OrdinalEncoder()
housing_cat_encoder = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoder[:10])
print(ordinal_encoder.categories_)

# Performing Data Cleaning using sklearn for categorical attributes
from sklearn.preprocessing import OneHotEncoder
onehot_Encoder = OneHotEncoder()
housing_cat_onehot = onehot_Encoder.fit_transform(housing_cat)
print(housing_cat_onehot.toarray())
print(onehot_Encoder.categories_)

# Creation of custom transformer to combine attributes
from sklearn.base import  BaseEstimator,TransformerMixin
room_ix, bedroom_ix, population_ix, households_ix=3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedroom_per_room=True):
        self.add_bedroom_per_room = add_bedroom_per_room
    def fit(self,x,y=None):
        return self
    def transform(self,X):
        rooms_per_household=X[:,room_ix]/X[:,households_ix]
        population_per_household=X[:, population_ix]/X[:,households_ix]
        if self.add_bedroom_per_room:
            bedroom_per_household=X[:,room_ix]/X[:,bedroom_ix]
            return np.c_[rooms_per_household,population_per_household,bedroom_per_household]
        else:
            return np.c_[rooms_per_household,population_per_household]
attr_adder = CombinedAttributesAdder()
housing_extra_attr = attr_adder.transform(housing.values)

# Transformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipline = Pipeline([("imputer", SimpleImputer(strategy="median")),                              # Piplines for numerical attributes
                        ("attribs_adder", CombinedAttributesAdder()),
                        ('std_scalar', StandardScaler())
                        ])
housing_num_tr = num_pipline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num",num_pipline,num_attribs),                                 # Piplines for specific columns
                                   ("cat", OneHotEncoder(), cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
