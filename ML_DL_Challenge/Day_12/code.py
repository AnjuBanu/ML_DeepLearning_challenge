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

# Creating a prediction model with Simple Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(housing_prepared,housing_label)
# Predicting few instances of training set to validate the model predictions
some_data = housing.iloc[:5]
some_labels = housing_label[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:::", lr_model.predict(some_data_prepared))
print("Actual:::" ,list(some_labels))
# Predicting the result with training data and Verifying the loss between the predicted and actual result
from sklearn.metrics import mean_squared_error
housing_predicted = lr_model.predict(housing_prepared)
mse=mean_squared_error(housing_label, housing_predicted)                                                # This results in underfitting model
print("MSE for LR:::",mse)
print("RMSE for LR:::",np.sqrt(mse))
# Predicting the result with complex models like decision tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_label)
housing_predicted = tree_reg.predict(housing_prepared)
mse=mean_squared_error(housing_label, housing_predicted)
print("MSE for Decision tree:::",mse)
print("RMSE for Decision tree:::",np.sqrt(mse))                                                          # This results in overfitting model
# Performing cross validation split to avoid overfitting of data
from sklearn.model_selection import cross_val_score
scores_dt = cross_val_score(tree_reg, housing_prepared, housing_label,
                         scoring="neg_mean_squared_error", cv=10)
scores_lr = cross_val_score(lr_model, housing_prepared, housing_label,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores_dt = np.sqrt(- scores_dt)
rmse_scores_lr = np.sqrt(- scores_lr)
def display_scores(scores):
    print("Scores:::", scores)
    print("Mean:::", scores.mean())
    print("Standard Deviation:::", scores.std())
display_scores(rmse_scores_dt)
display_scores(rmse_scores_lr)
# Predicting the result with Random forest
from sklearn.ensemble import RandomForestRegressor
forest_rg = RandomForestRegressor()
forest_rg.fit(housing_prepared, housing_label)
housing_predicted = forest_rg.predict(housing_prepared)
mse=mean_squared_error(housing_label, housing_predicted)
print("RMSE for Training data:::",mse)
scores_forest = cross_val_score(forest_rg, housing_prepared, housing_label,
                         scoring="neg_mean_squared_error", cv=10)
rmse_score_forest = np.sqrt(-scores_forest)
display_scores(rmse_score_forest)

# Saving the models as pickle file
import joblib
joblib.dump(forest_rg, "final_model.pkl")
# Loading the model
final_model = joblib.load("final_model.pkl")

# Fine tuning the models
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
               {'bootstrap':[False],'n_estimators':[3,10,15],'max_features':[2,3,4]}]
forest_rg = RandomForestRegressor()
grid_search = GridSearchCV(forest_rg, param_grid, cv=5,
                           scoring= "neg_mean_squared_error",
                           return_train_score=True)
grid_search.fit(housing_prepared,housing_label)
print(grid_search.best_params_)                                                                 # Identifying the bestg hypertunning paramters
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Analyzing the best models and their errors
feature_importance = grid_search.best_estimator_.feature_importances_
print(feature_importance)

# Evaluating the model with test set
final_model = grid_search.best_estimator_
X_test = test_set_strat.drop("median_house_value", axis=1)
y_test = test_set_strat["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_prediction = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_prediction)
print("Final RMSE:::", final_mse)

# Calculating 95% confidence interval for the predicted result
from scipy import  stats
confidence =0.95
squared_errors = (final_prediction - y_test)**2
values= stats.t.interval(confidence, len(squared_errors)-1,
                 loc=squared_errors.mean(),
                 scale = stats.sem(squared_errors))
print(np.sqrt(values))

