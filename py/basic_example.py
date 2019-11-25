#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: basic_example.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Gaussian Processes regression: basic introductory example
cf. https://scikit-learn.org
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)  # Fix the seed to get the same result


def f(x):
    """ The function to predict """
    return x * np.sin(x)


# --------------------------------------------------------------------------- #
# First the noiseless case
# --------------------------------------------------------------------------- #
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observation
y = f(X).flatten()

# Mesh the input space for evaluations of the real function
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate the Gaussian Process model
kernel = C(1., (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Liklihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the mesjed x-axis
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval
fig, ax = plt.subplots(figsize=((9, 6)))
ax.plot(x, f(x), 'r:', label=r'$f(x)=x sin(x)$')
ax.plot(X, y, 'r.', label='Observation')
ax.plot(x, y_pred, 'b-', label='Prediction')
ax.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
ax.legend(loc='best')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_xlim([0, 10])
plt.savefig('figs/noiseless.pdf')
plt.close()

# --------------------------------------------------------------------------- #
# Noisy case
# --------------------------------------------------------------------------- #
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observation and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(
    kernel=kernel, alpha=dy**2, n_restarts_optimizer=10)

# Fit to data using Maximum Liklihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the fuction, the prediction and the 95% confidence interval
fig, ax = plt.subplots(figsize=((9, 6)))
ax.plot(x, f(x), 'r:', label=r'$f(x)=x sin(x)$')
ax.errorbar(X.ravel(), y, dy, fmt='r.', label='Observation')
ax.plot(x, y_pred, 'b-', label='Prediction')
ax.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
ax.legend(loc='best')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_xlim([0, 10])
plt.savefig('figs/noise.pdf')
plt.close()
