# Importing libraries
import matplotlib.pyplot as plt
import ssl
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

ssl._create_default_https_context = ssl._create_unverified_context
tips = sns.load_dataset('tips')                                                                 # Loading the dataset for regression
X = tips.drop(columns='tip')
y = tips['tip']
np.random.seed(1)
test_frac = 0.25                                                                                # Manually splitting test and train data
test_size = int(len(y)*test_frac)
test_index = np.random.choice(np.arange(len(y)), test_size, replace=False)
X_train = X.drop(test_index)
y_train = y.drop(test_index)
X_test = X.loc[test_index]
y_test = y.loc[test_index]
X_train = pd.get_dummies(X_train, drop_first=True).reset_index(drop=True)                       # Performing one hot encoding
X_test = pd.get_dummies(X_test, drop_first=True).reset_index(drop=True)

# Bootstrapped tree creation and averaging the fitted values
class Bagger:
    def fit(self,X_train,y_train,B,max_depth=100,min_size=2,seed=None):
        self.X_train = X_train
        self.y_train = y_train
        self.N, self.D = X_train.shape
        self.B=B
        self.max_depth=max_depth
        self.min_size=min_size
        self.seed=seed
        self.trees=[]
        np.random.seed(seed)
        for bag in range(self.B):
            sample = list(np.random.choice(np.arange(self.N), 10, replace=True))
            X_train_bag = X_train.loc[sample]
            y_train_bag = y_train.reset_index(drop=True).loc[sample]
            tree = DecisionTreeRegressor(max_depth=self.max_depth,min_samples_split=self.min_size)
            tree.fit(X_train_bag,y_train_bag)
            self.trees.append(tree)

    def predict(self,X_test):
        y_hat = np.empty((len(self.trees), len(X_test)))
        for i,tree in enumerate(self.trees):
            y_hat[i] = tree.predict(X_test)
        return y_hat.mean(0)

i=0
j=0
fig, ax = plt.subplots(2,3,sharex=True, sharey=True,figsize=(7, 5))
for b in (np.arange(1,31,5)):
    bagger = Bagger()
    bagger.fit(X_train,y_train,B=b,max_depth = 20, min_size = 10, seed = 123)
    y_test_hat = bagger.predict(X_test)
    ## Plot
    ax[i,j].set(xlabel = r'$y$', ylabel = r'$\hat{y}$')
    ax[i,j].set_title(f"Bootstraps = {b}")
    sns.scatterplot(ax=ax[i,j], x=y_test, y=y_test_hat, color="g")
    if (j==2):
        i=i+1
        j=0
    else:
        j=j+1
plt.tight_layout()
plt.show()