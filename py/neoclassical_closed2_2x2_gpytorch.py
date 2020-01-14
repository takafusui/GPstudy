#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: neoclassical_closed2_2x2_gpytorch.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Neoclassical growth model in discrete time
Greenwood-Hercowitz-Huffman preferences
AR(1) TFP shock
"""
import sys
import warnings
import numpy as np
import torch
import gpytorch
sys.path.append('/Users/takafumi/my_src/PyTorch-LBFGS/functions/')
sys.path.append('/home/takafumi/my_src/PyTorch-LBFGS/functions/')
from LBFGS import FullBatchLBFGS
import pyipopt
# plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 15

print(r"GPyTorch is version {}".format(gpytorch.__version__))

np.random.seed(0)
torch.manual_seed(0)

# --------------------------------------------------------------------------- #
# Parameter setting
# --------------------------------------------------------------------------- #
A_bar = 1  # The mean of the technology shock
alpha = 0.36  # Capital share in the Cobb-Douglas production function
beta = 0.95  # Discount factor
psi = 0.25  # Coefficient of leisure in the utility function
theta = 1.5  # Elasticity of leisure in the utility function

# AR(1) technology shock
rho = 0.95  # Autocorrelation coefficient
mu = 0  # Mean
s = 0.01  # Standard deviation


# --------------------------------------------------------------------------- #
# Analytical solution
# --------------------------------------------------------------------------- #
def ls_compute(k, A=A_bar, alpha=alpha, psi=psi, theta=theta):
    """ Return the optimal labor supply """
    return (((1-alpha) * A * k**alpha) / (psi*theta))**(1 / (theta+alpha-1))


def kplus_compute_analytic(
        k, A=A_bar, alpha=alpha, beta=beta, psi=psi, theta=theta):
    """ Analytical solution
    Return the optimal capital stock in the next period """
    _ls = ls_compute(k, A, alpha, psi, theta)
    return alpha * beta * A * k**alpha * _ls**(1-alpha)


def c_compute_analytic(
        k, A=A_bar, alpha=alpha, beta=beta, psi=psi, theta=theta):
    """ Analytical solution
    Return the optimal consumption policy """
    _ls = ls_compute(k, A, alpha, psi, theta)
    return (1 - alpha*beta)*A*k**alpha*_ls**(1-alpha)


# --------------------------------------------------------------------------- #
# Gauss-Hermite quadrature
# --------------------------------------------------------------------------- #
# Nodes
x5 = np.sqrt(2) * s * np.array(
    [2.020182870456086, 0.9585724646138185, 0, -0.9585724646138185,
     -2.020182870456086]) + mu
# Weights
omega5 = np.pi**(-1/2) * np.array(
    [0.01995324205904591, 0.3936193231522412, 0.9453087204829419,
     0.3936193231522412, 0.01995324205904591])

print("GH5 nodes are {}".format(x5))
print("GH5 weights are {}".format(omega5))

# sys.exit(0)
# --------------------------------------------------------------------------- #
# Set the exogenous capital domain
# Must include the certainty equivalent steady state
# --------------------------------------------------------------------------- #
kbeg, kend = 0.05, 0.2  # Capital state
abeg, aend = 0.8, 1.2  # AR(1) technology state

# --------------------------------------------------------------------------- #
# Gaussian process, training and test dataset
# --------------------------------------------------------------------------- #
dim_input = 2  # Dimension of input
num_train = 10  # Number of training samples
num_test = 1000  # Number of test samples

# Training datasets
train_X = torch.stack([
    (kbeg - kend) * torch.rand(num_train) + kend,
    (abeg - aend) * torch.rand(num_train) + aend], dim=1)
train_shape = (num_train, dim_input)

# Test datasets, used to evaluate the approximation quality
test_X = torch.stack([
    (kbeg - kend) * torch.rand(num_test) + kend,
    (abeg - aend) * torch.rand(num_test) + aend], dim=1)
test_shape = (num_test, dim_input)

# Shape check
assert train_X.shape == train_shape, 'Shape is not {}'.format(train_shape)
assert test_X.shape == test_shape, 'Shape is not {}'.format(test_shape)

# sys.exit(0)


# --------------------------------------------------------------------------- #
# Initialize outputs
# Capital stock in the next period and the Lagrange multiplier
# --------------------------------------------------------------------------- #
def train_y_guess(X):
    """ Initialize training outputs """
    k = X[:, 0]  # Capital stock state
    a = X[:, 1]  # Technology shock state
    _train_y_kplus = k
    _train_y_ls = ls_compute(k=k, A=a, alpha=alpha, psi=psi, theta=theta)
    _train_y_c = a * k**alpha * _train_y_ls**(1 - alpha) - _train_y_kplus
    _train_y_lambd = 1 / (_train_y_c - psi * _train_y_ls**theta)
    return _train_y_kplus, _train_y_lambd


train_y_kplus, train_y_lambd = train_y_guess(train_X)
output_shape = (train_X.shape[0], )

# Shape check
assert train_y_kplus.shape == output_shape, 'Shape is not {}'.format(
    output_shape)
assert train_y_lambd.shape == output_shape, 'Shape is not {}'.format(
    output_shape)

# sys.exit(0)

# --------------------------------------------------------------------------- #
# Instantiate and initialize the Gaussian process
# --------------------------------------------------------------------------- #
# Define the Gaussian process model
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, trian_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --------------------------------------------------------------------------- #
# Train the model hyperparameters
# --------------------------------------------------------------------------- #
def TrainGPModel(train_X, train_y_kplus, train_y_lambd, training_iter,
                 print_status=False):
    """ 
    Train the Gaussian process and optimize the model hyperparameters
    """
    # Define the Gaussian process model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, trian_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_kplus = ExactGPModel(train_X, train_y_kplus, likelihood)
    gp_lambd = ExactGPModel(train_X, train_y_lambd, likelihood)

    print(gp_kplus)
    gp_kplus.train()
    gp_lambd.train()
    likelihood.train()

    # Use the LBFGS optimizer
    optimizer_kplus = FullBatchLBFGS(
        # Includes GaussianLikelihood parameters
        gp_kplus.parameters(), lr=0.01)
    optimizer_lambd = FullBatchLBFGS(
        # Includes GaussianLikelihood parameters
        gp_lambd.parameters(), lr=1)

    # Use the Adam
    # optimizer_kplus = torch.optim.Adam([
    #     # Includes GaussianLikelihood parameters
    #     {'params': gp_kplus.parameters()}, ], lr=learning_rate)
    # optimizer_lambd = torch.optim.Adam([
    #     # Includes GaussianLikelihood parameters
    #     {'params': gp_lambd.parameters()}, ], lr=learning_rate)

    # "Loss" for GPs - the marginal log likelihood
    mll_kplus = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, gp_kplus)
    mll_lambd = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, gp_lambd)

    def closure_kplus():
        """ Closure for kplus """
        # Zero gradients from previous iteration
        optimizer_kplus.zero_grad()
        # Output from model
        output_kplus = gp_kplus(train_X)
        # Calculate loss and backprop gradients
        loss_kplus = - mll_kplus(output_kplus, train_y_kplus)
        return loss_kplus

    def closure_lambd():
        """ Closure for lambda """
        # Zero gradients from previous iteration
        optimizer_lambd.zero_grad()
        # Output from model
        output_lambd = gp_lambd(train_X)
        # Calculate loss and backprop gradients
        loss_lambd = - mll_lambd(output_lambd, train_y_lambd)
        return loss_lambd

    loss_kplus = closure_kplus()
    loss_kplus.backward()
    loss_lambd = closure_lambd()
    loss_lambd.backward()

    # sys.exit(0)
    for i in range(training_iter):
        # Perform step and update curvature
        options_kplus = {
            'closure': closure_kplus, 'current_loss': loss_kplus,
            'max_ls': 10}
        loss_kplus, _, lr_kplus, _, F_eval_kplus, G_eval_kplus, _, \
            fail_kplus = optimizer_kplus.step(options_kplus)

        if print_status is True:
            print(
                'Iter %d/%d - Loss: %.3f - Lengthscale: %.3f - Noise: %.3f' % (
                    i + 1, training_iter, loss_kplus.item(),
                    gp_kplus.covar_module.base_kernel.lengthscale.item(),
                    gp_kplus.likelihood.noise.item()
                ))
        if fail_kplus:
            print('Convergence reached after {} iterations!'.format(i+1))
            
            break

    for i in range(training_iter):
        # Perform step and update curvature
        options_lambd = {
            'closure': closure_lambd, 'current_loss': loss_lambd,
            'max_ls': 10}
        loss_lambd, _, lr_lambd, _, F_eval_lambd, G_eval_lambd, _, \
            fail_lambd = optimizer_lambd.step(options_lambd)

        if print_status is True:
            print(
                'Iter %d/%d - Loss: %.3f - Lengthscale: %.3f - Noise: %.3f' % (
                    i + 1, training_iter, loss_lambd.item(),
                    gp_lambd.covar_module.base_kernel.lengthscale.item(),
                    gp_lambd.likelihood.noise.item()
                ))
        if fail_lambd:
            print('Convergence reached after {} iterations!'.format(i+1))
            break

    return gp_kplus, gp_lambd, likelihood

# Training
gp_kplus, gp_lambd, likelihood = TrainGPModel(
    train_X, train_y_kplus, train_y_lambd, training_iter=100)

# sys.exit(0)

# --------------------------------------------------------------------------- #
# Equilibrium conditions
# --------------------------------------------------------------------------- #
def euler(x0, state, gp_lambd):
    """
    Set of non-linear equilibrium conditions to be solved by IPOPT
    x0: Starting values for the optimization
    state: Current state, state[0]: capital state and state[1]: AR(1) TFP shock
    gp_lambd: Gaussian process regression model for lambda
    x[0]: Capital stock in the next time period
    x[1]: lambda, Lagrange multiplier associated with the budget constraint
    """
    nvar = 2  # Number of variables

    k, a = state[0], state[1]  # Extract the current state

    # All of plicies are assumed to be non-negative
    x_L = np.ones(nvar) * 1e-10
    x_U = np.ones(nvar) * np.inf

    # AR(1) technology shock
    # x5 is the Gauss-Hermite nodes
    aplus = np.empty_like(x5)
    for epsilon_idx, epsilon_plus in enumerate(x5):
        aplus[epsilon_idx] = a**rho * np.exp(epsilon_plus)

    # Current labor supply
    ls = ls_compute(k=k, A=a, alpha=alpha, psi=psi, theta=theta)

    def con(x, ls):
        """ Consumption """
        assert len(x) == nvar
        return 1 / x[1] + psi*ls**theta

    def ls_plus(x):
        """ Labor supply in the next period
        Discretize AR(1) shock by the Gauss-Hermite quadrature degree 5
        """
        assert len(x) == nvar
        _ls_plus = np.empty_like(aplus)
        for aplus_idx, aplus_val in enumerate(aplus):
            _ls_plus[aplus_idx] = ls_compute(
                k=x[0], A=aplus_val, alpha=alpha, psi=psi, theta=theta)
        return _ls_plus

    def lambd_plus(x):
        """ Return lambda (the Lagrange multiplier) in the next period
        Discretize AR(1) shock by the Gauss-Hermite quadrature degree 5
        """
        assert len(x) == nvar
        _lambdplus = np.empty_like(aplus)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for aplus_idx, aplus_val in enumerate(aplus):
                stateplus = torch.tensor([x[0], aplus_val])[None, :]
                observed_pred = gp_lambd(stateplus)
                _lambdplus[aplus_idx] = observed_pred.mean.numpy()
            return _lambdplus

    def eval_f(x):
        """ Dummy objective function """
        assert len(x) == nvar
        return 0

    def eval_grad_f(x):
        """ Gradient of the dummy objective function """
        assert len(x) == nvar
        grad_f = np.zeros(nvar)
        return grad_f

    ncon = nvar  # Number of constraints

    """ Complementarity constraints
    g0: Euler equation wrt. k_{t+1}
    g1: Resource constraint """

    g_L = np.zeros(ncon)
    g_U = g_L

    nnzj = int(nvar * ncon)  # Number of (possibly) non-zeros in Jacobian
    nnzh = int((nvar**2 - nvar) / 2 + nvar)  # Number of non-zeros in Hessian

    def eval_g(x):
        """ The system of non-linear equilibrium conditions """
        assert len(x) == nvar

        g0 = x[1] - np.sum(np.multiply(
            omega5,
            beta * alpha * lambd_plus(x) * aplus * x[0]**(
                alpha-1) * ls_plus(x)**(1-alpha)))
        g1 = a * k**alpha * ls**(1-alpha) - x[0] - con(x, ls)

        return np.array([g0, g1])

    def eval_jac_g(x, flag):
        """ Numerical approximation of the Jacobian of the system of
        non-linear equilibrium conditions
        Use the finite-difference-values option provided by IPOPT """
        assert len(x) == nvar
        # nvarncon = nvar * ncon
        # _eval_jac_g = np.empty(nvarncon)

        row_idx = np.empty(nnzj, dtype=int)  # Row index
        col_idx = np.empty(nnzj, dtype=int)  # Column index

        # Jacobian matrix structure
        if flag:
            for i in range(ncon):
                for j in range(nvar):
                    row_idx[j + i * nvar] = i
                    col_idx[j + i * nvar] = j

            return (row_idx, col_idx)

        # else:
        #     #  Finite Differences
        #     h = 1e-4
        #     gx1 = eval_g(x)

        #     for ixM in range(nvar):
        #         for ixN in range(ncon):
        #             xAdj = np.copy(x)
        #             xAdj[ixN] = xAdj[ixN]+h
        #             gx2 = eval_g(xAdj)
        #             _eval_jac_g[ixN + ixM * nvar] = (gx2[ixM] - gx1[ixM]) / h
        #     return _eval_jac_g

    # ----------------------------------------------------------------------- #
    # Define a NLP model
    # ----------------------------------------------------------------------- #
    pyipopt.set_loglevel(0)  # Let IPOPT quite

    neoclassical = pyipopt.create(
        nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f,
        eval_g, eval_jac_g)
    neoclassical.str_option("linear_solver", "ma57")
    neoclassical.str_option(
        "jacobian_approximation", "finite-difference-values")
    neoclassical.str_option("hessian_approximation", "limited-memory")
    neoclassical.num_option("tol", 1e-7)
    neoclassical.num_option("acceptable_tol", 1e-6)
    neoclassical.int_option("max_iter", 50)
    neoclassical.int_option("print_level", 3)

    xstar, zl, zu, constraint_multipliers, obj, status = neoclassical.solve(x0)

    if status not in [0, 1]:
        warnings.warn(
            "IPOPT fails to solve the system of non-linear equations. "
            "Use the starting value as the solution instead.")
        xstar = x0
    return xstar


# --------------------------------------------------------------------------- #
# Time iteration collocation
# --------------------------------------------------------------------------- #
def time_iter_gpr(num_train, num_test, training_iter):
    """
    Time iteration collocation with the Gaussian process regression
    num_train: Number of training examples
    dim_input: Dimension of inputs
    num_output: Number of outputs
    """
    num_iter = 500  # Number of time iterations
    epsilons = []  # Track the approximation error
    epsilon_tol = 1e-5  # Convergence tolrance

    # ----------------------------------------------------------------------- #
    # Generate a training dataset
    # ----------------------------------------------------------------------- #
    # Training datasets
    # train_X = torch.stack([
    #     torch.linspace(kbeg, kend, num_train),
    #     torch.linspace(abeg, aend, num_train)], dim=-1)
    train_X = torch.stack([
        (kbeg - kend) * torch.rand(num_train) + kend,
        (abeg - aend) * torch.rand(num_train) + aend], dim=1)

    # Initialize outputs
    train_y_kplus, train_y_lambd = train_y_guess(train_X)

    # Shape check
    assert train_X.shape == (num_train, 2), 'Shape is not (num_train, 2)'
    assert train_y_kplus.shape == (num_train, ), 'Shape is not (num_train, )'
    assert train_y_lambd.shape == (num_train, ), 'Shape is not (num_train, )'

    # Test datasets, used to evaluate the approximation quality
    test_X = torch.stack([
        torch.linspace(kbeg, kend, num_test),
        torch.linspace(abeg, aend, num_test)], dim=-1)

    print(r"Shape of the training dataset is {}".format(train_X.shape))
    print(r"Shape of the test dataset is {}".format(test_X.shape))

    # sys.exit(0)
    # ----------------------------------------------------------------------- #
    # Instantiate the Gaussian processes
    # ----------------------------------------------------------------------- #    
    gp_kplus, gp_lambd, likelihood = TrainGPModel(
        train_X, train_y_kplus, train_y_lambd, training_iter=100)

    # sys.exit(0)
    # ----------------------------------------------------------------------- #
    # Time iteration collocation
    # ----------------------------------------------------------------------- #
    for n in range(1, num_iter+1):

        # Get into evaluation (predictive posterior) mode
        gp_kplus.eval()
        gp_lambd.eval()
        likelihood.eval()

        # Starting value retliving from the previous optimization
        x0 = np.stack([train_y_kplus.numpy(), train_y_lambd.numpy()], axis=1)
        # print(x0)

        # Keep the optimal policies
        train_y_kplus = np.empty(num_train)
        train_y_lambd = np.empty(num_train)

        for idx, state in enumerate(train_X.numpy()):
            # For each state, solve the system of non-linear equations
            xstar = euler(x0[idx, :], state, gp_lambd)
            # Track the optimal policies
            train_y_kplus[idx] = xstar[0]
            train_y_lambd[idx] = xstar[1]

            print(xstar)
        print(type(train_y_kplus))
        print(train_y_lambd)
        # sys.exit(0)
        # ------------------------------------------------------------------- #
        # Train the Gaussian process with the optimal policy
        # ------------------------------------------------------------------- #
        # Training data
        train_y_kplus = torch.tensor(train_y_kplus, dtype=torch.float)
        train_y_lambd = torch.tensor(train_y_lambd, dtype=torch.float)

        # Training
        gp_kplus_updated, gp_lambd_updated, likelihood_updated = TrainGPModel(
            train_X, train_y_kplus, train_y_lambd, training_iter)

        # sys.exit(0)

        # ------------------------------------------------------------------- #
        # Approximation error analysis
        # Update the policy functions for the next iteration
        # ------------------------------------------------------------------- #
        # Switch to the evaluation mode
        gp_kplus.eval()
        gp_kplus_updated.eval()
        gp_lambd.eval()
        gp_lambd_updated.eval()
        likelihood.eval()
        likelihood_updated.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_kplus = likelihood(gp_kplus(test_X))
            observed_pred_lambd = likelihood(gp_lambd(test_X))
            observed_pred_update_kplus = likelihood_updated(
                gp_kplus_updated(test_X))
            observed_pred_update_lambd = likelihood_updated(
                gp_lambd_updated(test_X))

            epsilon = max(
                np.max(np.abs(observed_pred_kplus.mean.numpy()
                              - observed_pred_update_kplus.mean.numpy())),
                np.max(np.abs(observed_pred_lambd.mean.numpy()
                              - observed_pred_update_lambd.mean.numpy()))
                )
        epsilons.append(epsilon)  # Track the history of epsilon

        if n % 1 == 0:
            print('Iteration: {}, Approximation error: {:.3e}'.format(
                n, epsilons[-1]))

        if epsilons[-1] < epsilon_tol:
            # Terminate the time iteration and save the optimal surrogates
            gp_kplus_star = gp_kplus_updated
            gp_lambd_star = gp_lambd_updated
            print("Time iteration collocation is terminated successfuly with "
                  "{} iterations".format(n))
            break  # Terminate the iteration

        else:
            # Update the GP with the surrogates
            gp_kplus = gp_kplus_updated
            gp_lambd = gp_lambd_updated

        # sys.exit(0)

    return epsilons, gp_kplus_star, gp_lambd_star, likelihood


# --------------------------------------------------------------------------- #
# Compute the optimal policy functions
# --------------------------------------------------------------------------- #
epsilons, gp_kplus_star, gp_lambd_star, likelihood = time_iter_gpr(
    num_train=10, num_test=1000, training_iter=100)

# --------------------------------------------------------------------------- #
# Plot an approximation error
# --------------------------------------------------------------------------- #
plt.plot(range(len(epsilons)), epsilons)
plt.yscale('log')
plt.xlabel("Number of iterations")
plt.ylabel("Max approximation error")
plt.xlim([0, None])
plt.savefig('figs/GHH_2x2_epsilon.pdf')
plt.close()


# # --------------------------------------------------------------------------- #
# # Plot approximated policy functions
# # --------------------------------------------------------------------------- #
# gridplt = np.random.uniform([kbeg, abeg], [kend, aend], (250, dim_input))
# # Analytical solution when A = 1
# gridplt_analytic = np.random.uniform([kbeg, 1], [kend, 1], (50, dim_input))


# # Capital stock ------------------------------------------------------------- #
# gridplt_kplus = np.hstack([gridplt, np.ones_like(gridplt[:, 1:]) * 0])
# noise_dict_kplus = {'output_index': gridplt_kplus[:, 2:].astype(int)}
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     stateplus = torch.tensor([x[0], aplus_val])[None, :]
#                 observed_pred = gp_lambd(stateplus)
#                 _lambdplus[aplus_idx] = observed_pred.mean.numpy()

# kplus_star = gp_kplus_star.predict(gridplt_kplus, Y_metadata=noise_dict_kplus)[0]
# kplus_analytic = kplus_compute_analytic(gridplt_analytic[:, 0])

# fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
# ax.scatter(
#     gridplt[:, 0], gridplt[:, 1], kplus_star.ravel(), 'o', label='GP')
# ax.scatter(
#     gridplt_analytic[:, 0], gridplt_analytic[:, 1], kplus_analytic, 'o',
#     label='Analytic ($A_{t}=1$)')
# ax.set_xlabel(r"$K_t$")
# ax.set_ylabel(r"$A_t$")
# ax.set_zlabel(r"$K_{t+1}$")
# ax.set_zlim([0, None])
# ax.invert_xaxis()
# ax.legend(loc='best')
# plt.show()

# # Lambda -------------------------------------------------------------------- #
# gridplt_lambd = np.hstack([gridplt, np.ones_like(gridplt[:, 1:]) * 1])
# noise_dict_lambd = {'output_index': gridplt_lambd[:, 2:].astype(int)}
# lambd_star = gp_star.predict(gridplt_lambd, Y_metadata=noise_dict_lambd)[0]

# fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
# ax.scatter(
#     gridplt[:, 0], gridplt[:, 1], lambd_star.ravel(), 'o')
# ax.set_xlabel(r"$K_t$")
# ax.set_ylabel(r"$A_t$")
# ax.set_zlabel(r"$\lambda_{t}$")
# ax.invert_xaxis()
# plt.show()

# # Labor supply -------------------------------------------------------------- #
# ls_star = ls_compute(
#     gridplt[:, 0], gridplt[:, 1], alpha=alpha, psi=psi, theta=theta)

# # Consumption --------------------------------------------------------------- #
# c_star = 1 / lambd_star.ravel() + psi * ls_star**theta

# c_star_analytic = c_compute_analytic(
#     gridplt_analytic[:, 0], A=gridplt_analytic[:, 1], alpha=alpha, beta=beta, 
#     psi=psi, theta=theta)

# fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
# ax.scatter(
#     gridplt[:, 0], gridplt[:, 1], c_star, 'o', label='GP')
# ax.scatter(
#     gridplt_analytic[:, 0], gridplt_analytic[:, 1], c_star_analytic, 'o',
#     label='Analytic ($A_{t}=1$)')
# ax.set_xlabel(r"$K_t$")
# ax.set_ylabel(r"$A_t$")
# ax.set_zlabel(r"$C_{t}$")
# ax.invert_xaxis()
# ax.legend(loc='best')
# plt.show()


# # ### Mean and the standard deviation

# # In[20]:


# # 7x7 cartesian grid
# cart_size = 7j
# k_mgrid, a_mgrid = np.mgrid[kbeg:kend:cart_size, abeg:aend:cart_size]

# # kplus
# kplus_mu = np.empty_like(k_mgrid)
# kplus_var = np.empty_like(k_mgrid)
# # lambda
# lambd_mu = np.empty_like(k_mgrid)
# lambd_var = np.empty_like(k_mgrid)
# for k_idx, k in enumerate(k_mgrid[:, 0]):
#     for a_idx, a in enumerate(a_mgrid[0, :]):
#         # kplus
#         state = np.array([[k, a, 0]])
#         noise_dict = {'output_index': state[:, 2:].astype(int)}
#         kplus_mu[k_idx, a_idx] = gp.predict(state, Y_metadata=noise_dict)[0]
#         kplus_var[k_idx, a_idx] = gp.predict(state, Y_metadata=noise_dict)[1]
#         # lambda
#         state = np.array([[k, a, 1]])
#         noise_dict = {'output_index': state[:, 2:].astype(int)}
#         lambd_mu[k_idx, a_idx] = gp.predict(state, Y_metadata=noise_dict)[0]
#         lambd_var[k_idx, a_idx] = gp.predict(state, Y_metadata=noise_dict)[1]

# # Standard deviation
# kplus_sigma = np.sqrt(kplus_var)
# lambd_sigma = np.sqrt(lambd_var)

# fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
# ax.scatter(
#     k_mgrid, a_mgrid, kplus_mu, 'o', label=r'$\mu$')
# ax.scatter(
#     k_mgrid, a_mgrid, kplus_mu+2*kplus_sigma, 'o', label=r'$\mu + 2\sigma$')
# ax.scatter(
#     k_mgrid, a_mgrid, kplus_mu-2*kplus_sigma, 'o', label=r'$\mu - 2\sigma$')
# ax.set_xlabel(r"$K_t$")
# ax.set_ylabel(r"$A_t$")
# ax.set_zlabel(r"$K_{t+1}$")
# ax.invert_xaxis()
# ax.legend(loc='best')

# plt.show()

# fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})
# ax.scatter(
#     k_mgrid, a_mgrid, lambd_mu, 'o', label=r'$\mu$')
# ax.scatter(
#     k_mgrid, a_mgrid, lambd_mu+2*lambd_sigma, 'o', label=r'$\mu + 2\sigma$')
# ax.scatter(
#     k_mgrid, a_mgrid, lambd_mu-2*lambd_sigma, 'o', label=r'$\mu - 2\sigma$')
# ax.set_xlabel(r"$K_t$")
# ax.set_ylabel(r"$A_t$")
# ax.set_zlabel(r"$\lambda_{t}$")
# ax.invert_xaxis()
# ax.legend(loc='best')

# plt.show()
