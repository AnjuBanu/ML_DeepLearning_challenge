![img.png](img.png)

## About
Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.

The purpose of this project is not to produce as optimized and computationally efficient algorithms as possible but rather to present the inner workings of them in a transparent and accessible way.

---
### Day 1

Ordinary Linear Regression:

Linear Regression is a relatively simple method that is extremely widely-used. It is also a great stepping stone for more sophisticated methods, making it a natural algorithm to study first. A regression analysis is simply a method of estimating the relationship between a dependent variable and a set of independent variables.

Have understood and implemented the Linera regression by using a concept of minimizing the loss and maximizing the likelihood.

![img.png](ML_DL_Challenge/Day_1/img.png)

---
### Day 2

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In machine learning, we use gradient descent to update the parameters of our model.
I have implementated a simple gradient descent to understand the behavior based on different learning rates and momentum

![img.png](ML_DL_Challenge/Day_2/img_1.png) 
![img.png](ML_DL_Challenge/Day_2/img_2.png)
![img.png](ML_DL_Challenge/Day_2/Learning_rate.png)

---
### Day 3

Ridge and Lasso are two of the most important regularization techniques in linear regression. Both of them solve the problems of plain-old LR by introducing some bias leading to lower variance and thus avoiding overfitting. The hyperparameter lambda controls how severe the regularization is for features with different magnitudes.

I have understood the math behind the these regularization techniques and implemented them and presented there score, MSA and MSE.

![img.png](ML_DL_Challenge/Day_3/code.png) 
![img.png](ML_DL_Challenge/Day_3/output.png)
![img.png](ML_DL_Challenge/Day_3/plot.png)

---
### Day 4

Classifier is a supervised learning algorithm thats attemps to identify an observation belongs to one or more groups. The most commonly used discriminative and generative classification models are Logistic Regression, Perceptron Algorithm and Fisher's Linear Discriminant.

I have understood the model structure and parameters estimiation for all these algorithms and implemented them by training different datasets.

![img.png](ML_DL_Challenge/Day_4/code.png)

---
### Day 5

A Discriminative model as we saw in previous post models the decision boundary between the classes. A Generative Model ‌explicitly models the actual distribution of each class. In final both of them is predicting the conditional probability P(Animal | Features). But Both models learn different probabilities. A Generative Model ‌learns the joint probability distribution p(x,y). It predicts the conditional probability with the help of Bayes Theorem. A Discriminative model ‌learns the conditional probability distribution p(y|x).

I have gained an insight into Joint Probability vs Conditional Probability, Generative VS Discriminative Models and Naive bayes algorithm and compared it with discriminative models LDA and QDA.


![img.png](ML_DL_Challenge/Day_5/code.png)
![img.png](ML_DL_Challenge/Day_5/plot1.png)
![img.png](ML_DL_Challenge/Day_5/plot2.png)
![img.png](ML_DL_Challenge/Day_5/plot3.png)

---
### Day 6

Decision Trees are a non-parametric supervised learning method used for classification and regression. Decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model. Decision tree builds classification or regression models in the form of a tree structure. Decision trees can handle both categorical and numerical data.

I have demonstrated how to fit regression and decision trees and also gained deep understanding on Hyperparameters like Size Regulation, Minimum Reduction in RSS and Pruning (Pre and post) and other key factors like Entropy, Information gain, Impurity of a node and various algorithms likes ID3, CART, C4.5

![img.png](ML_DL_Challenge/Day_6/code.png)

---
### Day 7

Tree Ensemble Bagging - Due to their high variance, decision trees often fail to reach a level of precision comparable to other predictive algorithms. Ensemble methods combine the output of multiple simple models, often called “learners”, in order to create a final model with lower variance. Bagging, short for bootstrap aggregating, combines the results of several learners trained on bootstrapped samples of the training data. The process of bagging is very simple yet often quite powerful.

For a given dataset created a bagging model which trains many learners on bootstrapped samples of the training data and aggregates the results into one final model and have also outlined the behaviour for different number of bootstraps each time fitting a decision tree regressor.

![img.png](ML_DL_Challenge/Day_7/code.png)
![img.png](ML_DL_Challenge/Day_7/plot.png)

---
### Day 8

When we are learning Machine learning it is best to experiment with real world data and not artificial datasets. I have used "California housing price" dataset here and have explored the dataset in details and gained an understanding on framing the problem based on he required business objective and also perceived other notions like pipelining (Sequence of Data processing components), selection of performance measure line RMSE, MSE, downloading the data, getting a quick look of the data structure and simple creation of test and train dataset manually.

![img.png](ML_DL_Challenge/Day_8/code.png)
![img.png](ML_DL_Challenge/Day_8/plot_bin.png)
![img.png](ML_DL_Challenge/Day_8/report.png)

---
### Day 9

The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model. It is a fast and easy procedure to perform, the results of which allow you to compare the performance of machine learning algorithms for your predictive modeling problem.

It can be used for classification or regression problems and can be used for any supervised learning algorithm. I have performed various train-test split procedure like cyclic redundancy check, spli based on identifier column and also other methods like StratifiedShuffleSplit (skewed data) and train_test_split by using sklearn library and verified the proportion of test randomly picked matches with the proportion of actual dataset.

![img.png](ML_DL_Challenge/Day_9/code.png)
![img.png](ML_DL_Challenge/Day_9/comparison.png)
![img.png](ML_DL_Challenge/Day_9/income_cat.png)

---
### Day 10

Data visualization is the representation of data or information in a graph, chart, or other visual format. It communicates relationships of the data with images. This is important because it allows trends and patterns to be more easily seen and visual summary of information makes it easier to identify patterns and trends than looking through thousands of rows on a spreadsheet. It’s the way the human brain works.

The goal is to go into little more depth and I have tried to discover and visualize the data to gain insight. Visualized the geographical data based on different attributes by spotting the patterns. Also Verified the correlation between the attributes and transformed them to get better insight.

![img.png](ML_DL_Challenge/Day_10/code.png)
![img.png](ML_DL_Challenge/Day_10/img1.png)
![img.png](ML_DL_Challenge/Day_10/img2.png)
![img.png](ML_DL_Challenge/Day_10/img3.png)
![img.png](ML_DL_Challenge/Day_10/corr.png)

---
### Day 11

Data cleansing or data cleaning is the process of detecting and correcting corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.

In order to really seek to understanding on Data cleasing, have performed various activities like performing data cleaning manually and also by using libraries. Have understood the design principles like "Estimators", "Transformers" and "Predictors". Performed various encoding techniques on numerical and Categorical attributes and executed a simple pipeline for tranformation of attributes.

![img.png](ML_DL_Challenge/Day_11/code1.png)
![img.png](ML_DL_Challenge/Day_11/code2.png)

---
### Day 12

For any given machine learning problem, numerous algorithms can be applied and multiple models can be generated. Having a wealth of options is good, but deciding on which model to implement in production is crucial. Though we have a number of performance metrics to evaluate a model, it’s not wise to implement every algorithm for every problem. This requires a lot of time and a lot of work. So it’s important to know how to select the right algorithm for a particular task.

By looking at the factors of the dataset, I have select 3 different models and refined my selection based on accuracy scores. I have also understood the various techniques to fine tune the model like "Grid search" and "Randomized Search" and identified the best hyper parameter tuning parameters and used the final model to evaluate the test set.

![img.png](ML_DL_Challenge/Day_12/code1.png)
![img.png](ML_DL_Challenge/Day_12/code2.png)

---
### Day 13

Classification is the process of recognizing, understanding, and grouping ideas and objects into preset categories or “sub-populations.” Using pre-categorized training datasets, machine learning programs use a variety of algorithms to classify future datasets into categories. Classification algorithms in machine learning use input training data to predict the likelihood that subsequent data will fall into one of the predetermined categories.

I have achieved understanding on various helper function and performed a simple Binary Image classification by using some simple classifiers like SGD and measured accuracy using cross validation technique and also made aware that impact of skewed data on accuracy performance measure.

![img.png](ML_DL_Challenge/Day_13/code.png)
