#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: jacobian_torch_numpy.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Compute Jacobian with numpy and PyTorch
"""
import sys
import numpy as np
import torch

nvar = 3
ncon = 3


def eval_g(x):
    g0 = x[0] * x[1] + x[2]
    g1 = x[1] + x[0] * x[2]
    g2 = torch.sin(x[2])

    # if flag:
    #     return [g0, g1, g2]
    # else:
    return [g0, g1, g2]


nnzj = int(nvar * ncon)  # Number of elements in the Jacobian matrix


def eval_jac_g(x, flag):
    """ Numerical approximation of the Jacobian of the system of
    non-linear equilibrium conditions
    Use the finite-difference-values option provided by IPOPT """
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
        x_tensor = x.clone().detach().requires_grad_(True)
        jac = []
        for i in range(ncon):
            grad, = torch.autograd.grad(eval_g(x_tensor)[i], x_tensor)
            jac.append(grad)
        return torch.stack(jac).flatten().numpy()


# --------------------------------------------------------------------------- #
# Some checks
# --------------------------------------------------------------------------- #
x = torch.tensor([1., 2., 3.], dtype=torch.float64)  # On this point
print('Function is {}'.format(np.array(eval_g(x))))
print('Structure of Jacobian is {}'.format(eval_jac_g(x, True)))
print('Elements of Jacobian are {}'.format(eval_jac_g(x, False)))

# Gradient check
x_tensor = x.clone().detach().requires_grad_(True)
chk = torch.autograd.gradcheck(eval_g, x_tensor)
print(chk)
