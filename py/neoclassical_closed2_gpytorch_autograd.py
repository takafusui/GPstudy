#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: neoclassical_closed2_gpytorch_autograd.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Neoclassical growth model in discrete time
Greenwood-Hercowitz-Huffman preferences
GPyTorch: Gaussian process regression
PyTorch: automatic gradient
IPOPT + pyipopt: optimizer
Output is z-scored
"""
import sys
import warnings
import numpy as np
import torch
import gpytorch
import pyipopt
from scipy.optimize import root  # Find the steady state
# plot
import matplotlib.pyplot as plt
from matplotlib import rc
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
plt.rcParams["font.size"] = 15

np.random.seed(123)

print("PyTorch version is {}".format(torch.__version__))
print("GPyTorch version is {}".format(gpytorch.__version__))

# --------------------------------------------------------------------------- #
# Parameter setting
# No TFP shock, deterministic model
# --------------------------------------------------------------------------- #
A = 1  # Technology level
alpha = 0.36  # Capital share in the Cobb-Douglas production function
beta = 0.95  # Discount factor
psi = 0.25  # Coefficient of leisure in the utility function
theta = 1.5  # Elasticity of leisure in the utility function


# --------------------------------------------------------------------------- #
# Analytical solution
# --------------------------------------------------------------------------- #
def ls_compute(k, A=A, alpha=alpha, psi=psi, theta=theta):
    """ Return the optimal labor supply """
    return (((1-alpha) * A * k**alpha) / (psi*theta))**(1 / (theta+alpha-1))


def kplus_compute_analytic(
        k, A=A, alpha=alpha, beta=beta, psi=psi, theta=theta):
    """ Analytical solution
    Return the optimal capital stock in the next period """
    _ls = ls_compute(k, A, alpha, psi, theta)
    return alpha * beta * A * k**alpha * _ls**(1-alpha)


def c_compute_analytic(k, A=A, alpha=alpha, beta=beta, psi=psi, theta=theta):
    """ Analytical solution
    Return the optimal consumption policy """
    _ls = ls_compute(k, A, alpha, psi, theta)
    return (1 - alpha*beta)*A*k**alpha*_ls**(1-alpha)


def k_infty_compute(k_infty):
    """ Compute the stationary point in capital """
    return k_infty - kplus_compute_analytic(k_infty)


# Find the stationary point
res = root(k_infty_compute, x0=1, method='hybr')
print("Stationary point is {:5f}".format(res.x[0]))

# sys.exit(0)
# --------------------------------------------------------------------------- #
# Plot the initial guess
# --------------------------------------------------------------------------- #
kbeg, kend = 0.05, 0.2
grid_plt = 250
kgrid_plt = np.linspace(kbeg, kend, grid_plt)

# Plot the kplus policy function
plt.plot(kgrid_plt, kplus_compute_analytic(kgrid_plt))
plt.plot(kgrid_plt, kgrid_plt, 'k:')
plt.xlabel(r'$K_{t}$')
plt.ylabel(r'$K_{t+1}$')
plt.xlim([kbeg, kend])
plt.ylim([0, None])
plt.savefig('../figs/GHH_kplus_analytic.pdf')
plt.close()

# Plot the consumption policy function
plt.plot(kgrid_plt, c_compute_analytic(kgrid_plt))
plt.xlabel(r'$K_{t}$')
plt.ylabel(r'$C_{t}$')
plt.xlim([kbeg, kend])
plt.ylim([0, None])
plt.savefig('../figs/GHH_c_analytic.pdf')
plt.close()

# sys.exit(0)
# --------------------------------------------------------------------------- #
# Gaussian process, training and test dataset
# --------------------------------------------------------------------------- #
num_train = 25  # Number of training samples
num_test = 1000  # Number of test samples

# Training datasets, assuming a uniform distribution
train_x = torch.linspace(kbeg, kend, num_train, dtype=torch.float64)
train_y_kplus = train_x

# Test datasets
test_x = torch.linspace(kbeg, kend, num_test, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# Train the model hyperparameters
# --------------------------------------------------------------------------- #
def TrainGPModel(
        train_X, train_y_kplus, learning_rate, training_iter, print_skip):
    """
    Train the Gaussian process and optimize the model hyperparameters
    Use the torch.optim.Adam optimizer
    train_X.shape == [n, d]
    train_y_kplus.shape == [n]
    """
    # ----------------------------------------------------------------------- #
    # Instantiate and initialize the Gaussian process
    # ----------------------------------------------------------------------- #
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
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

    # ----------------------------------------------------------------------- #
    # Find the optimal model hyperparameters
    # ----------------------------------------------------------------------- #
    gp_kplus.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam([
        # Includes GaussianLikelihood parameters
        {'params': gp_kplus.parameters()}, ], lr=learning_rate)

    # Loss for GP - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_kplus)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = gp_kplus(train_X)
        # Calculate the  loss and backprop gradients
        loss = - mll(output, train_y_kplus)
        loss.backward()
        if print_skip != 0 and (i+1) % print_skip == 0:
            print(
                'Iter %d/%d - Loss: %.3f - lengthscale: %.3f - noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    gp_kplus.covar_module.base_kernel.lengthscale.item(),
                    gp_kplus.likelihood.noise.item()
                )
            )
        optimizer.step()

    return gp_kplus, likelihood


# --------------------------------------------------------------------------- #
# Z-scored and train the Gaussian process
# --------------------------------------------------------------------------- #
def mean_std(train_y):
    """
    Compute the mean and the standard deviation of the output train_y
    train_y: Original output (torch.tensor)
    return mean (torch.tensor) and standard deviation (torch.tensor)
    """
    return train_y.mean(-1, keepdim=True), train_y.std(-1, keepdim=True)


def z_scored(train_y):
    """
    Standardization (z-scored)
    train_y: Original output (torch.tensor)
    return zscored output (torch.tensor)
    """
    train_y_mean, train_y_std = mean_std(train_y)
    return (train_y - train_y_mean) / train_y_std


def scale_back(train_y, train_y_zscored):
    """
    Scale back to the original output
    train_y: Original output (torch.tensor)
    train_y_zscored: z-scored output (torch.tensor)
    return scaled-backed output (torch.tensor)
    """
    train_y_mean, train_y_std = mean_std(train_y)
    return train_y_zscored * train_y_std + train_y_mean


train_y_kplus_zscored = z_scored(train_y_kplus)
gp_kplus, likelihood = TrainGPModel(
    train_x, train_y_kplus_zscored, learning_rate=0.1, training_iter=1000,
    print_skip=50)

# sys.exit(0)
# --------------------------------------------------------------------------- #
# Plot the posterior distirubtion with the hyperparameters optimization
# --------------------------------------------------------------------------- #
# Get into evaluation (predictive posterior) mode
gp_kplus.eval()
likelihood.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    plot_x = torch.linspace(kbeg, kend, 50, dtype=torch.float64)
    observed_pred_kplus = likelihood(gp_kplus(plot_x))

# sys.exit(0)
with torch.no_grad():
    # Initialize plot for kplus
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred_kplus.confidence_region()
    lower = scale_back(train_y_kplus, lower)
    upper = scale_back(train_y_kplus, upper)
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y_kplus.numpy(), 'k*')
    # Plot predictive means as blue line
    observed_pred_kplus = scale_back(train_y_kplus, observed_pred_kplus.mean)
    ax.plot(plot_x.numpy(), observed_pred_kplus.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(plot_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_xlim([kbeg, kend])
    ax.set_ylim([0, None])
    ax.set_xlabel(r"$K_{t}$")
    ax.set_ylabel(r"$K_{t+1}$")
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    plt.savefig('../figs/GHH_kplus_after.pdf')
    plt.close()

# sys.exit(0)


# --------------------------------------------------------------------------- #
# Equilibrium conditions
# --------------------------------------------------------------------------- #
def euler(x0, k, gp_kplus, likelihood, train_y_org):
    """
    Set of a non-linear equilibrium condition solved by IPOPT + pyipopt
    Jacobian is supplied via the automatic gradient of PyTorch
    Hessian is approximated by the IPOPT's limited memory option
    x0: Starting values for the optimization
    k: Current capital state
    gp_kplus, likelihood: Gaussian process regression model (GPyTorch)
    train_y_org: Original traininig outputs
    x[0]: Capital stock in the next period
    """

    nvar = 1  # Number of a variable

    # All of plicies are assumed to be non-negative
    x_L = np.zeros(nvar, dtype=np.float64)
    x_U = np.ones(nvar, dtype=np.float64) * 1000

    def eval_f(x):
        """ Dummy objective function """
        assert len(x) == nvar
        return 0

    def eval_grad_f(x):
        """ Gradient of the dummy objective function """
        assert len(x) == nvar
        grad_f = np.zeros(nvar, dtype=np.float64)
        return grad_f

    def ls_compute(k):
        """ Return the optimal labor supply """
        return (((1-alpha) * A * k**alpha) / (psi*theta))**(
            1 / (theta+alpha-1))

    # Labor supply today
    ls = ls_compute(k)

    ncon = nvar  # Number of a constraint

    """ Equilibrium condition
    g0: Euler equation wrt. k_{t+1}
    """

    g_L = np.zeros(ncon, dtype=np.float64)
    g_U = g_L

    nnzj = int(nvar * ncon)  # Number of (possibly) non-zeros in Jacobian
    nnzh = int((nvar**2 - nvar) / 2 + nvar)  # Number of non-zeros in Hessian

    def eval_g_tensor(x):
        """
        The system of non-linear equilibrium conditions
        PyTorch version
        """
        assert len(x) == nvar

        # Consumption today
        con = A * k**alpha * ls**(1-alpha) - x[0]

        # Labor supply tomorrow
        ls_plus = ls_compute(x[0])

        # Capital stock in the day after tomorrow
        if type(x) is torch.Tensor:
            k_plusplus = likelihood(gp_kplus(x[None, :][0])).mean
        elif type(x) is np.ndarray:
            k_plusplus = likelihood(gp_kplus(torch.tensor(
                [x[0]], dtype=torch.float64)[None, :])).mean
        else:
            raise TypeError("x shold be either torch.Tensor or np.ndarray")

        # Scale back to the original output range
        k_plusplus = scale_back(train_y_org, k_plusplus)

        # Consumption tomorrow
        con_plus = A * x[0]**alpha * ls_plus**(1-alpha) - k_plusplus

        # ------------------------------------------------------------------- #
        # Euler equation
        # ------------------------------------------------------------------- #
        g0 = 1/con - beta * alpha * 1/con_plus * A * x[0]**(alpha-1) \
            * ls_plus**(1-alpha)

        return [g0]

    def eval_g_numpy(x):
        """ Convert from Tensor to numpy so that IPOPT can handle """
        return np.array(eval_g_tensor(x), dtype=np.float64)

    def eval_jac_g(x, flag):
        """
        Jacobian of the system of non-linear equilibrium conditions
        Automatic differenciation provided by PyTorch
        """
        assert len(x) == nvar

        row_idx = np.empty(nnzj, dtype=int)  # Row index
        col_idx = np.empty(nnzj, dtype=int)  # Column index

        # Jacobian matrix structure
        if flag:
            for i in range(ncon):
                for j in range(nvar):
                    row_idx[j + i * nvar] = i
                    col_idx[j + i * nvar] = j

            return (row_idx, col_idx)

        else:
            # Automatic gradient by PyTorch
            assert len(x) == nvar
            _x = torch.tensor(x, requires_grad=True)
            jac = []
            for i in range(ncon):
                grad, = torch.autograd.grad(eval_g_tensor(_x)[i], _x)
                jac.append(grad)
            return torch.stack(jac).flatten().numpy()

    # ----------------------------------------------------------------------- #
    # Define a NLP model
    # ----------------------------------------------------------------------- #
    pyipopt.set_loglevel(0)  # Let IPOPT quite

    neoclassical = pyipopt.create(
        nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f,
        eval_g_numpy, eval_jac_g)
    neoclassical.str_option("linear_solver", "ma57")
    neoclassical.str_option("hessian_approximation", "limited-memory")
    # neoclassical.str_option("derivative_test", "first-order")  # Pass!
    neoclassical.int_option("print_level", 1)

    xstar, zl, zu, constraint_multipliers, obj, status = neoclassical.solve(x0)

    if status not in [0, 1]:
        warnings.warn(
            "IPOPT fails to solve the system of non-linear equations. "
            "Use the starting value as the solution instead.")
        xstar = x0
    return xstar


# --------------------------------------------------------------------------- #
# Time iteration collocation with the Gaussian process regression
# --------------------------------------------------------------------------- #
def time_iter_gp(num_train, training_iter):
    """
    Time iteration collocation with the Gaussian process regression
    num_train: Number of training examples
    training_iter: Number of training iterations
    """
    num_iter = 500  # Maximum number of time iterations
    epsilons = []  # Track the approximation error
    epsilon_tol = 1e-6  # Convergence tolrance

    # ----------------------------------------------------------------------- #
    # Generate a training dataset, uniform distribution
    # ----------------------------------------------------------------------- #
    train_x = torch.linspace(kbeg, kend, num_train, dtype=torch.float64)
    train_y_kplus = train_x  # Initial guess
    # Z-scored
    train_y_kplus_zscored = z_scored(train_y_kplus)
    # ----------------------------------------------------------------------- #
    # Instantiate and train the Gaussian processes
    # ----------------------------------------------------------------------- #
    gp_kplus, likelihood = TrainGPModel(
        train_x, train_y_kplus_zscored, learning_rate=0.1, training_iter=1000,
        print_skip=50)

    # sys.exit(0)
    # ----------------------------------------------------------------------- #
    # Time iteration collocation
    # ----------------------------------------------------------------------- #
    for n in range(1, num_iter+1):
        # Get into evaluation (predictive posterior) mode
        gp_kplus.eval()
        likelihood.eval()

        # Keep the original value range
        train_y_kplus_org = train_y_kplus

        # Starting value retliving from the previous optimization
        x0 = train_y_kplus.numpy()[:, None]

        # Track the optimal policies
        train_y_kplus = np.empty_like(train_y_kplus)

        for idx, k in enumerate(train_x.numpy()):
            # For each state, solve the system of non-linear equations
            xstar = euler(x0[idx], k, gp_kplus, likelihood, train_y_kplus_org)
            # Track the optimal policies
            train_y_kplus[idx] = xstar
            # sys.exit(0)
        # ------------------------------------------------------------------- #
        # Train the Gaussian process with the optimal policy
        # ------------------------------------------------------------------- #
        # Training data before taking z-score
        train_y_kplus = torch.from_numpy(train_y_kplus)
        # Z-scored output
        train_y_kplus_zscored = z_scored(train_y_kplus)
        # Training
        gp_kplus_updated, likelihood_updated = TrainGPModel(
            train_x, train_y_kplus_zscored, learning_rate=0.1,
            training_iter=1000, print_skip=0)

        # ------------------------------------------------------------------- #
        # Approximation error analysis
        # Update the policy functions for the next iteration
        # ------------------------------------------------------------------- #
        # Switch to the evaluation mode
        gp_kplus.eval()
        gp_kplus_updated.eval()
        likelihood.eval()
        likelihood_updated.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(gp_kplus(test_x))
            pred_update = likelihood_updated(gp_kplus_updated(test_x))

            epsilon = np.max(np.abs(
                pred.mean.numpy() - pred_update.mean.numpy()))
        epsilons.append(epsilon)  # Track the history of epsilon

        if n % 1 == 0:
            print('Iteration: {}, Approximation error: {:.3e}'.format(
                n, epsilons[-1]))

        if epsilons[-1] < epsilon_tol:
            # Terminate the time iteration and save the optimal surrogates
            gp_kplus_star = gp_kplus_updated
            likelihood_star = likelihood_updated
            train_y_kplus_org = train_y_kplus_org
            print("Time iteration collocation is terminated successfuly with "
                  "{} iterations".format(n))
            break  # Terminate the iteration

        else:
            # Update the GP with the surrogates
            gp_kplus = gp_kplus_updated
            likelihood = likelihood_updated

        # sys.exit(0)

    return epsilons, gp_kplus_star, likelihood_star, train_y_kplus_org


# --------------------------------------------------------------------------- #
# num_train = 30, training_iter = 250
# --------------------------------------------------------------------------- #
epsilons, gp_kplus_star, likelihood_star, train_y_kplus_org \
    = time_iter_gp(25, 250)

# sys.exit(0)

# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
# Approximation error
plt.plot(epsilons)
plt.yscale('log')
plt.xlabel("Number of iterations")
plt.ylabel("Max approximation error")
plt.xlim([0, None])
plt.savefig('../figs/GHH_epsilon.pdf')
plt.close()

# Turn on the evaluation mode to plot the policy functinos
gp_kplus_star.eval()
likelihood_star.eval()

# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    plot_x = torch.linspace(kbeg, kend, 50, dtype=torch.float64)
    pred = likelihood_star(gp_kplus_star(plot_x))
    kplus_star_mu = pred.mean  # Z-scored value
    kplus_star_mu = scale_back(train_y_kplus_org, kplus_star_mu)
    kplus_lower, kplus_upper = pred.confidence_region()  # Z-scored value
    kplus_lower = scale_back(train_y_kplus_org, kplus_lower)
    kplus_upper = scale_back(train_y_kplus_org, kplus_upper)

with torch.no_grad():
    # Initialize plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # Plot the predicted mean
    ax.plot(plot_x.numpy(), kplus_star_mu, 'k-', label=r'Mean')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(plot_x.numpy(), kplus_lower.numpy(), kplus_upper.numpy(),
                    alpha=0.4, label=r'Confidence')
    ax.set_xlim([kbeg, kend])
    ax.set_ylim([0, None])
    ax.set_xlabel(r"$K_{t}$")
    ax.set_ylabel(r"$K_{t+1}$")
    ax.legend(loc='best')
    plt.savefig('../figs/GHH_kplus_star.pdf')
    plt.close()
