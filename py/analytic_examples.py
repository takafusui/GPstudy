#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: analytic_example.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Gaussian Processes regression: analytic examples
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 15

np.random.seed(1)  # Fix the seed to get the same result


# --------------------------------------------------------------------------- #
# The function to be predicted and the true values
# --------------------------------------------------------------------------- #
def f(x):
    """ The function to be predicted """
    return np.abs(0.25 - x[:, 0]**2 - x[:, 1]**2)


def g(x):
    """ The function to be predicted """
    return np.exp(0.3 * x[:, 0] + 0.7 * x[:, 1])


def h(x):
    """ The function to be predicted """
    return 1 / (np.abs(0.5 - x[:, 0]**4 - x[:, 1]**4) + 0.1)


num_test = 1000
num_input = 2
X_test_f = np.random.uniform(0., 1., (num_test, num_input))
X_test_h = X_test_f
X_test_g = np.random.uniform([-1, -1], [1, 1], (num_test, num_input))
y_true_f = f(X_test_f)
y_true_g = g(X_test_g)
y_true_h = h(X_test_h)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
ax.plot(X_test_f[:, 0], X_test_f[:, 1], y_true_f.ravel(), 'o', markersize=3,
        label=r'True')
ax.set_xlabel(r"$x$")
ax.set_xlim([0., 1.])
ax.set_ylabel(r"$y$")
ax.set_ylim([0., 1.])
ax.set_zlabel(r"$f\left(x, y\right)$")
ax.legend(loc='best')
plt.savefig('figs/analytic_f.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
ax.plot(X_test_g[:, 0], X_test_g[:, 1], y_true_g.ravel(), 'o', markersize=3,
        label=r'True')
ax.set_xlabel(r"$x$")
ax.set_xlim([-1., 1.])
ax.set_ylabel(r"$y$")
ax.set_ylim([-1., 1.])
ax.invert_yaxis()
ax.set_zlabel(r"$g\left(x, y\right)$")
ax.set_zlim([0., 3.])
ax.legend(loc='best')
plt.savefig('figs/analytic_g.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
ax.plot(X_test_h[:, 0], X_test_h[:, 1], y_true_h.ravel(), 'o', markersize=3,
        label=r'True')
ax.set_xlabel(r"$x$")
ax.set_xlim([0., 1.])
ax.set_ylabel(r"$y$")
ax.set_ylim([0., 1.])
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_zlabel(r"$h\left(x, y\right)$")
ax.set_zlim([0., 10.])
ax.legend(loc='best')
plt.savefig('figs/analytic_h.pdf')
plt.close()


# --------------------------------------------------------------------------- #
# Compute the approximation error
# --------------------------------------------------------------------------- #
def compute_mse(y_true, y_pred):
    """ Compute the mean squared error"""
    return 1 / len(y_true) * np.sum(np.square(y_true - y_pred.ravel()))


# --------------------------------------------------------------------------- #
# Check the approximation quality wrt. the numper of training points
# --------------------------------------------------------------------------- #
num_trainings_f = [10, 55, 100, 550, 1000]
num_trainings_g = [4, 8, 16, 32, 64, 128]
num_trainings_h = [10, 55, 100, 550, 1000]


def return_mse(
        analyticFunc, X_test, y_true, num_trainings, num_input, alpha=1e-8):
    mse_err = []  # Track the MSE error
    for num in num_trainings:
        # Generate training data
        X_train = np.random.uniform(-1., 1., (num, num_input))
        y_train = analyticFunc(X_train).reshape(num, 1)
        # Instantiate the Gaussian Process model
        kernel = RBF() * ConstantKernel(1e-20, (1e-25, 1e-15))
            # + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 10))
        # kernel = C() * RBF(1, (1e-5, 1e5))
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-8, n_restarts_optimizer=10,
            normalize_y=False)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(X_train, y_train)
        # Make the prediction
        y_pred, sigma = gp.predict(X_test, return_std=True)
        # Track the error
        mse_err.append(compute_mse(y_true, y_pred))

    return mse_err


# --------------------------------------------------------------------------- #
# Compute and plot MSE
# --------------------------------------------------------------------------- #
mse_err_f = return_mse(f, X_test_f, y_true_f, num_trainings_f, num_input)
mse_err_g = return_mse(g, X_test_g, y_true_g, num_trainings_g, num_input)
mse_err_h = return_mse(h, X_test_h, y_true_h, num_trainings_h, num_input)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(num_trainings_f, mse_err_f)
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Mean squared error')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([np.min(num_trainings_f), np.max(num_trainings_f)])
plt.savefig('figs/mse_f_analytic.pdf')
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(num_trainings_g, mse_err_g)
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Mean squared error')
ax.set_yscale('log')
ax.set_xlim([np.min(num_trainings_g), np.max(num_trainings_g)])
plt.savefig('figs/mse_g_analytic.pdf')
plt.close()


fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(num_trainings_h, mse_err_h)
ax.set_xlabel('Number of training examples')
ax.set_ylabel('Mean squared error')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([np.min(num_trainings_h), np.max(num_trainings_h)])
plt.savefig('figs/mse_h_analytic.pdf')
plt.close()
