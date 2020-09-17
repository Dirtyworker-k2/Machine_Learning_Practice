'''
@Descripttion: 
@version: 
@Author: sch
@Date: 2020-07-16 11:35:03
@LastEditors: sch
@LastEditTime: 2020-07-18 11:36:10
'''
import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib


def load_data(path, transpose=True):
    data = sio.loadmat(path)

    X = data.get('X')
    y = data.get('y').ravel()

    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
    
    return X, y


def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    
    cax = ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.Oranges,
                interpolation='sinc')
    fig.colorbar(cax)
    ax.set_xticklabels(np.array([]))
    ax.set_yticklabels(np.array([]))



path = '/Users/mac/Desktop/ML/Neural Networks/ex3data1.mat'
X, y = load_data(path)



def plot_100_images(X):
    sample_idx = np.random.choice(np.arange(X.shape[0]) ,100)
    sample_images = X[sample_idx, :]

    size = int(np.sqrt(X.shape[1]))
    
    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True,
                            figsize=(8, 8))
    
    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(sample_images[r * 10 + c].reshape((size, size)), 
                                cmap=matplotlib.cm.Oranges)
            ax[r, c].set_xticklabels(np.array([]))
            ax[r, c].set_yticklabels(np.array([]))