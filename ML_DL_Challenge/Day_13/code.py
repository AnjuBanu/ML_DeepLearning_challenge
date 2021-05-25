from sklearn.datasets import fetch_openml
mnist = fetch_openml (mnist_784, version=1)
print(mnist.keys())												# Importing MNSIT dataset			
X = mnist[data]												# Independent features
y = mnist[target]												# Dependent features
print(X.shape)
print(y.shape)

import matplotlib.pyplot as plt
some_digit = X[0]
print(X[0].shape)		
some_digit_updated = some_digit.reshape(28,28)					# Reshaping the flattnened matriz to 28X28
print(some_digit_updated.shape)								

plt.imshow(some_digit_updated, cmap='binary')					# Plotting the first test labels using binary values
plt.axis(off)
plt.show()
print(y[0] == 5)

# Converting the Y labels from string to numbers
import numpy as np 
y = y.astype(np.int64)

# Splitting the test and train data set
X_train, X_test, y_train, y_test = X[6000],X[6000],
								   y[6000],y[6000]

# Training using a binary classifier (SGD)
from sklearn.linear_model import SGDClassifier
y_train5 = (y_train == 5)
y_test5 = (y_test == 5)
sgd_model = SGDClassifier(random_state=123)
sgd_model.fit(X_train,y_train5)
sgd_model.predict([some_digit])   

# Implementing own cross validation function
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfold = StratifiedKFold(n_splits=3, shuffle=True, 
                         random_state = 42)
for train_index, test_index in skfold.split(X_train,y_train)
  cln_sgd_model = clone(sgd_model)
  X_train_folds = X_train[train_index]
  y_train_folds = y_train5[train_index]
  X_test_folds = X_test[test_index]
  y_test_folds = y_test5[test_index]
  cln_sgd_model.fit(X_train_folds,y_train_folds)
  y_pred = cln_sgd_model.predict(X_test_folds)
  n_correct = sum(y_pred == y_test_folds)	
  print(n_correct  len(y_pred))								# Verifying the accuracy for every split


# Using cross validation from sklearn
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_model, X_train,y_train5, cv=3, 
                scoring=accuracy)

# Understanding impact of Accuracy in skewed dataset
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator)
  def fit (self, X, y=None)
    return self
  
  def predict (self, X_test)
    return np.zeros((len(X_test),1), dtype=bool)
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train5, cv =3, 
                scoring = accuracy)
