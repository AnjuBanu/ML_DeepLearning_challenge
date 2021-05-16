# Importing Packages
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

# Loading the dataset
ssl._create_default_https_context = ssl._create_unverified_context
tips = sns.load_dataset('tips')                                             # Loading the dataset for regression
X = tips.drop(columns='tip')
y = tips['tip']
penguins = sns.load_dataset('penguins')                                     # Loading the dataset for classification
penguins = penguins.dropna().reset_index(drop=True)
X_n = penguins.drop(columns = 'species')
y_n = penguins['species']

# Regression tree
np.random.seed(1)
test_frac = 0.25                                                            # Manually splitting test and train data
test_size = int(len(y)*test_frac)
test_index = np.random.choice(np.arange(len(y)), test_size, replace=False)
X_train = X.drop(test_index)
y_train = y.drop(test_index)
X_test = X.loc[test_index]
y_test = y.loc[test_index]
X_train = pd.get_dummies(X_train, drop_first=True)                          # Performing one hot encoding
X_test = pd.get_dummies(X_test, drop_first=True)
dtr = DecisionTreeRegressor(max_depth=7,min_samples_split=5)                # Setting the hyper-tuning parameters
dtr.fit(X_train,y_train)                                                    # Fitting a model
y_predict = dtr.predict(X_test)                                             # Predicting the test data
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(y_test, y_predict)
ax.set(xlabel=r'$y$',ylabel=r'$\hat{y}$',title=r'$y$ vs. $\hat{y}$')
plt.show()

# Classification tree
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y_n)*test_frac)
test_index = np.random.choice(np.arange(len(y_n)), test_size, replace=False)
X_train = X_n.drop(test_index)
y_train = y_n.drop(test_index)
X_test = X_n.loc[test_index]
y_test = y_n.loc[test_index]
X_train = pd.get_dummies(X_train, drop_first=True)                          # Performing one hot encoding
X_test = pd.get_dummies(X_test, drop_first=True)
dtc = DecisionTreeClassifier(max_depth=5,min_samples_split=6)               # Setting the hyper-tuning parameters
dtc.fit(X_train,y_train)                                                    # Fitting a model
y_predict = dtc.predict(X_test)                                             # Predicting the test data
print(np.mean(y_test == y_predict))# Importing Packages
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier

# Loading the dataset
ssl._create_default_https_context = ssl._create_unverified_context
tips = sns.load_dataset('tips')                                             # Loading the dataset for regression
X = tips.drop(columns='tip')
y = tips['tip']
penguins = sns.load_dataset('penguins')                                     # Loading the dataset for classification
penguins = penguins.dropna().reset_index(drop=True)
X_n = penguins.drop(columns = 'species')
y_n = penguins['species']

# Regression tree
np.random.seed(1)
test_frac = 0.25                                                            # Manually splitting test and train data
test_size = int(len(y)*test_frac)
test_index = np.random.choice(np.arange(len(y)), test_size, replace=False)
X_train = X.drop(test_index)
y_train = y.drop(test_index)
X_test = X.loc[test_index]
y_test = y.loc[test_index]
X_train = pd.get_dummies(X_train, drop_first=True)                          # Performing one hot encoding
X_test = pd.get_dummies(X_test, drop_first=True)
dtr = DecisionTreeRegressor(max_depth=7,min_samples_split=5)                # Setting the hyper-tuning parameters
dtr.fit(X_train,y_train)                                                    # Fitting a model
y_predict = dtr.predict(X_test)                                             # Predicting the test data
fig,ax = plt.subplots(figsize=(7,5))
sns.scatterplot(y_test, y_predict)
ax.set(xlabel=r'$y$',ylabel=r'$\hat{y}$',title=r'$y$ vs. $\hat{y}$')
plt.show()

# Classification tree
np.random.seed(1)
test_frac = 0.25
test_size = int(len(y_n)*test_frac)
test_index = np.random.choice(np.arange(len(y_n)), test_size, replace=False)
X_train = X_n.drop(test_index)
y_train = y_n.drop(test_index)
X_test = X_n.loc[test_index]
y_test = y_n.loc[test_index]
X_train = pd.get_dummies(X_train, drop_first=True)                          # Performing one hot encoding
X_test = pd.get_dummies(X_test, drop_first=True)
dtc = DecisionTreeClassifier(max_depth=5,min_samples_split=6)               # Setting the hyper-tuning parameters
dtc.fit(X_train,y_train)                                                    # Fitting a model
y_predict = dtc.predict(X_test)                                             # Predicting the test data
print(np.mean(y_test == y_predict))