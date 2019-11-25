#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: multi_dims.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:

"""
import sys
import numpy as np
from matplotlib import pyplot as plt
# import cPickle as pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 15

# Fix the seed
np.random.seed(1)


# --------------------------------------------------------------------------- #
# Test function
# --------------------------------------------------------------------------- #
def f(x):
    """ The 2D test function to be predicted """
    return np.sin(x[:, 0]) * np.cos(x[:, 1])


# Generate training data
n_sample = 100  # Number of training points
dim = 2  # Dimensions

X = np.random.uniform(-1., 1., (n_sample, dim))
y = f(X).reshape(n_sample, 1) + np.random.randn(n_sample, 1) * 0.005

# --------------------------------------------------------------------------- #
# Instantiate a Gaussian Process model
# --------------------------------------------------------------------------- #
# kernel = RBF()
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis / training points
y_pred, sigma = gp.predict(X, return_std=True)


# --------------------------------------------------------------------------- #
# Compute MSE
# --------------------------------------------------------------------------- #
n_sample_test = 50
X_test1 = np.random.uniform(-1., 1., (n_sample_test, dim))
y_pred1, sigma1 = gp.predict(X_test1, return_std=True)
squared_err1 = np.sum(np.square(np.abs(y_pred1 - y[:50])))
mse1 = squared_err1 / len(y_pred1)
print("The MSE is {:e}".format(mse1))

# --------------------------------------------------------------------------- #
# Generate the second training data and evaluate an approximation error
# --------------------------------------------------------------------------- #
n_test = 50
dim = 2
X_test2 = np.random.uniform(-1., 1., (n_test, dim))
y_pred2, sigma2 = gp.predict(X_test2, return_std=True)
squared_err2 = np.sum(np.square(np.abs(y_pred2 - y[:50])))
mse2 = squared_err2 / len(y_pred2)
print("The MSE is {:e}".format(mse2))

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
ax.plot(X[:, 0], X[:, 1], y.ravel(), 'o', label=r'True')
ax.plot(X_test1[:, 0], X_test1[:, 1], y_pred1.ravel(), 'o', c='red',
        markersize=5, label=r'Predicted 1')
ax.plot(X_test2[:, 0], X_test2[:, 1], y_pred2.ravel(), 'o', c='orange',
        markersize=5, label=r'Predicted 2')

ax.invert_yaxis()
ax.legend(loc='best')
plt.savefig('figs/multi_dims.pdf')
plt.close()

sys.exit(0)
