import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Implementation of Linear Regression
class LinearRegression:
    # Identifying the Data required and its dimensions
    def fit (self, X, y, intercept = False):
        if intercept == False:
            ones = np.ones(len(X)).reshape(len(X) ,1)
            X = np.concatenate((ones ,X), axis=1)
        self.X = np.array(X)
        self.y = np.array(y)
        self.N ,self.D = X.shape
        # Estimate the parame
        XtX = np.dot(self.X.T, self.X)
        XtXin = np.linalg.inv(XtX)
        xty = np.dot(self.X.T, self.y)
        self.beta_hat = np.dot(XtXin, xty)
        # Predict the values
        self.yhat = np.dot(self.X, self.beta_hat)
        # Calculate the loss function
        self.l = (0.5) * np.sum((self.y - self.yhat )* *2)
    # To create predictions
    def predict (self, x_test, intercept = True):
        self.test_yhat = np.dot(x_test, self.beta_hat)

# Importing the dataset boston
boston = datasets.load_boston()
X = boston['data']                  # Independent features
y = boston['target']                # Dependent features
model = LinearRegression()          # Instantiating the model
model.fit(X,y)                      # Fitting the model


fig, ax = plt.subplots()            # Plotting the actual and predicted values
sns.scatterplot(y ,model.yhat)
ax.set_xlabel("y", size = 25)
ax.set_ylabel(r"$\hat{y}$", size = 25, rotation = 0, labelpad = 25)
ax.set_title("$y$ vs $\hat{y}$",  size = 25)
sns.despine()
