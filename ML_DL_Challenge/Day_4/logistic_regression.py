# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Importing the datasets
cancer_data = datasets.load_breast_cancer()                                     # Binary dataset
wine_data = datasets.load_wine()                                                # Multinomial dataset
X_cancer = cancer_data['data']
y_cancer = cancer_data['target']
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_cancer,y_cancer,
                                                    test_size=0.1,
                                                    random_state=123)
X_wine = wine_data['data']
y_wine = wine_data['target']
X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_wine,y_wine,
                                                    test_size=0.1,
                                                    random_state=123)
# Binary Logistic regression
binary_model = LogisticRegression(C=1000, max_iter=100000)                      # Parameters to converge
binary_model.fit(X_c_train,y_c_train)
binary_y_hats = binary_model.predict(X_c_test)                                  # Prediction for each class
binary_p_hats = binary_model.predict_proba(X_c_test)                            # Probability for all classes
print (binary_model.score(X_c_train,y_c_train))
print (binary_model.score(X_c_test,y_c_test))

# Multiclass Logistic regression
multi_model = LogisticRegression(multi_class="multinomial",                     # Parameters to conve
                                 C=1000,
                                 max_iter=100000)
multi_model.fit(X_w_train,y_w_train)
print (multi_model.score(X_w_train,y_w_train))
print (multi_model.score(X_w_test,y_w_test))

# Perceptron Algorithm
perc_model=Perceptron()                                                         # Important Algorithm for Neural network
perc_model.fit(X_c_train,y_c_train)
print (perc_model.score(X_c_train,y_c_train))
print (perc_model.score(X_c_test,y_c_test))

# Fisher's Linear Discriminant
lds_model=LinearDiscriminantAnalysis(n_components=1)                            # Reducing the data to one dimension
lds_model.fit(X_c_train,y_c_train)
f0=np.dot(X_c_train,lds_model.coef_[0])[y_c_train == 0]
f1=np.dot(X_c_train,lds_model.coef_[0])[y_c_train == 1]
fig,ax=plt.subplots(figsize=(7,5))
sns.distplot(f0,bins=25,kde=False,color="red",label="class1")                   # Histogram plot to identify classes
sns.distplot(f1,bins=25,kde=False,color="blue",label="class2")
ax.set_xlabel(r'$f\hspace{.25}(x_n)$', fontsize=10)
ax.set_title("Histogram of $f\hspace{.25}(x_n)$ by class", fontsize = 10)
ax.legend()
plt.show()