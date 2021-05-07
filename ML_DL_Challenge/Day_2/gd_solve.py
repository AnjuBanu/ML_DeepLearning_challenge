import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def gradient_descent (max_iteration, threshold, w_init, obj_func, w_func, learning_rate, momentum):
    w = w_init
    w_history = w;
    f_history = obj_func(w)
    delta = np.zeros(w.shape)
    i=0
    thr = 1.0e10
    while i< max_iteration and threshold<thr:
        delta = -learning_rate*(w_func(w))+momentum*delta   # Calculating the learning rate score
        w=w+delta                                           # Increasing or decreasing the slope value
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w)))
        i+=1
        thr = np.absolute(f_history[-1] - f_history[-2])
    return w_history, f_history

def visualize_fw():
    xcoord = np.linspace(-10.0,10.0,50)
    ycoord = np.linspace(-10.0, 10.0, 50)
    w1,w2 = np.meshgrid(xcoord,ycoord)
    pts  = np.vstack((w1.flatten(), w2.flatten()))          # Function value at each point
    pts = pts.transpose()                                   # All 2D points on the grid
    f_vals = np.sum(pts*pts, axis =1)
    plt.show()
    return pts, f_vals

# Helper function to annotate a single point
def annotate_pt(text,xy,xytext,color):
    plt.plot(xy[0],xy[1],marker = 'P',markersize=10, c=color)
    plt.annotate(text, xy = xy,
                       xytext = xytext,
                       arrowprops = dict(arrowstyle = '->',
                       color = color,
                       connectionstyle = 'arc3'))

def function_plot (pts, f_vals):
    f_plot = plt.scatter (pts[:,0], pts[:,1],
                          c=f_vals,
                          vmin=min(f_vals),
                          vmax=max(f_vals),
                          cmap = "viridis")
    plt.colorbar(f_plot)
    annotate_pt('global minima',(0,0), (-5,-7), "yellow")


# Objective function
def f(w,extra=[]):
    return np.sum(w*w)

# Function to compute the gradient
def grad(w,extra=[]):
    return 2*w

pts, f_vals = visualize_fw()

def visualize_learning(w_history):                          # Make the function plot
    function_plot(pts, f_vals)                              # Plot the history
    plt.plot(w_history[:, 0], w_history[:, 1], marker='o', c='magenta')
    annotate_pt('min found',                                # Annotate the point found at last iteration
                (w_history[-1, 0], w_history[-1, 1]),
                (-1, 7), 'Red')
    iter = w_history.shape[0]
    for w, i in zip(w_history, range(iter - 1)):            # Annotate with arrows to show history
        plt.annotate("",
                     xy=w, xycoords='data',
                     xytext=w_history[i + 1, :], textcoords='data',
                     arrowprops=dict(arrowstyle='<-',
                                     connectionstyle='angle3'))
# Function to plot the objective function
# and learning history annotated by arrows
# to show how learning proceeded
def sol_gd():
    rand = np.random.RandomState(19)
    w_init = rand.uniform(-10,10,2)
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(18, 12))
    max_iteration = 5
    threshold = -1
    learning_rate = [0.05,0.2,0.5,0.8]
    momentum = [0,0.5,0.9]
    ind = 1
    for val in momentum:
        for eta,col in zip (learning_rate, [0,1,2,3]):
            plt.subplot(3,4,ind)
            w_history, f_history = gradient_descent(max_iteration, threshold, w_init, f, grad, eta, val)
            visualize_learning(w_history)
            ind = ind+1
            plt.text(-9,12,'Learning rate=' +str(eta), fontsize=13)
            if col == 1:
                plt.text(10, 15, 'Momentum=' + str(eta), fontsize=13)
    fig.subplots_adjust(hspace=0.5, wspace=.3)
    plt.show()

sol_gd()
