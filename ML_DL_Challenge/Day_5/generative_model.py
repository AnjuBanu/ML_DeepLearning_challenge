# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Importing the dataset
wine = datasets.load_wine()

# Creating both discriminative and generative models
X,y= wine['data'], wine['target']
lda = LinearDiscriminantAnalysis()                                  # Linear discriminative analysis mode
lda.fit(X,y)
qda = QuadraticDiscriminantAnalysis()                               # Quadratic discriminative analysis model
qda.fit(X,y)
gnb = GaussianNB()                                                  # Naive Bayes model (Generative)
gnb.fit(X,y)

# Generate graphs to view the decision boundaries
def graph_model(X,model,model_name,n0=1000,n1=1000,figsize=(7,5),label_every=4):
    # Generate X for plotting
    d0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), n0)
    d1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), n1)
    X_plot = np.array(np.meshgrid(d0_range, d1_range)).T.reshape(-1, 2)

    # Get class predictions
    y_plot = model.predict(X_plot).astype(int)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(y_plot.reshape(n0, n1).T,
                cmap=sns.color_palette('Pastel1', 3),
                cbar_kws={'ticks': sorted(np.unique(y_plot))})
    xticks, yticks = ax.get_xticks(), ax.get_yticks()
    ax.set(xticks=xticks[::label_every], xticklabels=d0_range.round(2)[::label_every][0:8],
           yticks=yticks[::label_every], yticklabels=d1_range.round(2)[::label_every][0:7])
    ax.set(xlabel='X1', ylabel='X2', title=model_name + ' Predictions by X1 and X2')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.show()

X_2d=X.copy()[:,2:4]
lda_2d = LinearDiscriminantAnalysis()                                  # Linear discriminative analysis mode
lda_2d.fit(X_2d,y)
graph_model(X_2d,lda_2d,"LDA")
qda_2d = QuadraticDiscriminantAnalysis()                               # Quadratic discriminative analysis model
qda_2d.fit(X_2d,y)
graph_model(X_2d,qda_2d,"QDA")
gnb_2d = GaussianNB()                                                  # Naive Bayes model (Generative)
gnb_2d.fit(X_2d,y)
graph_model(X_2d,gnb_2d,"Naive Bayes")


