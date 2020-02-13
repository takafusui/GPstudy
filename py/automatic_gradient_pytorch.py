#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: automatic_gradient_pytorch.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Pytorch discussion forums
https://discuss.pytorch.org/t/automatic-gradient-and-torch-sum/68861
"""
import sys
import numpy as np
import torch

print(r"PyTorch version is {}".format(torch.__version__))

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
k = 0.1
a = 1

alpha = 0.36
beta = 0.95
psi = 0.25
theta = 1.5
s = 0.01
mu = 0
rho = 0.95

# Gauss-Hermite quadrature
# Nodes
x5 = np.sqrt(2) * s * torch.tensor(
    [2.020182870456086, 0.9585724646138185, 0, -0.9585724646138185,
     -2.020182870456086], dtype=torch.float64) + mu

# Weights
omega5 = np.pi**(-1/2) * torch.tensor(
    [0.01995324205904591, 0.3936193231522412, 0.9453087204829419,
     0.3936193231522412, 0.01995324205904591], dtype=torch.float64)

nvar = 1  # Number of variables
ncon = nvar  # Number of constraints

nnzj = int(nvar * ncon)  # Number of (possibly) non-zeros in Jacobian


def ls_compute(k, A, alpha=alpha, psi=psi, theta=theta):
    """ Return the optimal labor supply """
    return (((1-alpha) * A * k**alpha) / (psi*theta))**(1 / (alpha+theta-1))


# Current labor supply
ls = ls_compute(k=k, A=a, alpha=alpha, psi=psi, theta=theta)

# AR(1) shock
aplus = torch.empty(x5.shape, dtype=torch.float64)
for epsilon_idx, epsilon_plus in enumerate(x5):
    aplus[epsilon_idx] = a**rho * np.exp(epsilon_plus)


# sys.exit(0)
# --------------------------------------------------------------------------- #
# Functions
# eval_g returns the set of constraints that needs to be first-derivative
# eval_grad_f returns (1) the structure of the Jacobian and (2) its values
# --------------------------------------------------------------------------- #
def eval_g(x):
    """ The system of non-linear equilibrium conditions
    x[0]: Capital stock in the next period
    """

    # Consumption today
    con = a * k**alpha * ls**(1-alpha) - x[0]

    # Labor supply tomorrow
    ls_plus = torch.empty(x5.shape, dtype=torch.float64)
    for aplus_idx, aplus_val in enumerate(aplus):
        ls_plus[aplus_idx] = ls_compute(
            k=x[0], A=aplus_val, alpha=alpha, psi=psi, theta=theta)
    # Capital stock tomorrow
    k_plusplus = torch.empty(aplus.shape, dtype=torch.float64)
    for aplus_idx, aplus_val in enumerate(aplus):
        k_plusplus[aplus_idx] = torch.tensor(k)  # Needs to be modified

    # Consumption tomorrow
    con_plus = aplus * x[0]**alpha * ls_plus**(1-alpha) - k_plusplus

    # ----------------------------------------------------------------------- #
    # Euler equation
    # ----------------------------------------------------------------------- #
    g0 = 1 / con - beta * alpha * torch.sum(omega5 * (
        1 / con_plus * aplus * x[0]**(alpha-1) * ls_plus**(1-alpha)))

    return [g0]


# Test
print(r"Constraint returns {}".format(eval_g(np.array([k]))))
# sys.exit(0)


def eval_jac_g(x, flag):
    """ Numerical approximation of the Jacobian of the system of
    non-linear equilibrium conditions
    Jacobian is computed by the automatic gradient of PyTorch """
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
        x_tensor = torch.from_numpy(x).requires_grad_(True)
        jac = []
        for i in range(ncon):
            grad, = torch.autograd.grad(eval_g(x_tensor)[i], x_tensor)
            jac.append(grad)
        return torch.stack(jac).flatten().numpy()


# Test
print(r"The structure of the Jacobian is {}".format(
    eval_jac_g(np.array([k]), True)))
print(r"The values of each element in the Jacobian are {}".format(
    eval_jac_g(np.array([k]), False)))

k_tensor = torch.from_numpy(np.array([k])).requires_grad_(True)
chk = torch.autograd.gradcheck(eval_g, k_tensor)
print('Is the gradient check passed? {}'.format(chk))
