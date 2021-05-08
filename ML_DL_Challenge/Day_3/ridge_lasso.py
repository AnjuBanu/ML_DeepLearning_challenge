# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,LassoCV,BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn import datasets

# Importing the dataset boston
i=0
boston = datasets.load_boston()
X = boston['data']                                              # Independent features
y = boston['target']                                            # Dependent feature
x_train,x_test,y_train,y_test=train_test_split(X,y,             # Creating train and test split
                                            test_size=0.1,
                                            random_state=123)

model_list = {"Linear"  : LinearRegression(),                   # Creating dict for different models
              "Ridge1"  : Ridge(alpha=0.01),
              "RidgeCV" : RidgeCV(alphas=np.arange(1, 100, 5),
                                 scoring='r2',
                                 cv=10),
              "Lasso"   : Lasso(),
              "LassoCV" : LassoCV(alphas=np.arange(0.000000001, 1, 0.05),
                                  cv=10)}

fig, axes = plt.subplots(5, sharex=True, sharey=True, figsize=(15,8))
fig.suptitle('Y versus Yhat')
i=0
print ("-----------------------------------------------------------------")
print ("model_name\t\tTrain Score\t\tTest Score\t\tMAE\t\t\tMSE")
print ("-----------------------------------------------------------------")
for key, model in model_list.items():                              # Fitting and predicting using different models
    model_reg = model
    model_fit = model_reg.fit(x_train,y_train)
    model_pred = model_reg.predict(x_test)
    train_score = round(model.score(x_train, y_train),3)
    test_score = round(model.score(x_test, y_test),3)
    msa = round(mean_absolute_error(y_test, model_pred),3)
    mse = round(mean_squared_error(y_test, model_pred),3)
    print(f"{key}\t\t\t{train_score}\t\t\t{test_score}\t\t\t{msa}\t\t{mse}")
    axes[i].set_title(key,fontsize = 8,loc='left')                 # Plotting for very model
    sns.scatterplot(ax=axes[i], x=y_test, y=model_pred)
    i=i+1

plt.tight_layout()
plt.show()