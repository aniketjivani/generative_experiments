# bayes optimization under uncertainty.\n",
# Based on the paper: Beland and Nair (NIPS 2017) \"Bayesian Optimization under Uncertainty\"\n",
# Conduct all experiments with BoTorch"
# With some help from Claude and Github Copilot : ) 

#%% Import libraries

import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import botorch
import os
import math

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf


# %%
BRANIN_MEAN = torch.tensor([[-math.pi / 2, 12.275], [math.pi / 2, 2.275], [9.42477796, 2.475]])

def branin(x):
    x1, x2 = x[:, 0], x[:, 1]
    a = 1.0
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * math.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
    return y
# %%

def branin_noise(x, delta):
    x1, x2 = x[:, 0], x[:, 1]
    x1_noise = x1 + delta[:, 0]
    x2_noise = x2 + delta[:, 1]
    x_noise = torch.stack([x1_noise, x2_noise], dim=-1)
    y = branin(x_noise)
    return y


# %%

def initialize_model(train_x, train_y, state_dict=None):
    train_x = train_x.clone().detach()
    train_y = train_y.clone().detach()
    model = SingleTaskGP(train_x, train_y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # nice touch to allow importing of pre-trained model.
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acq_func(acq_func, bounds, n_restarts=10, raw_samples=20):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=n_restarts,
        raw_samples=raw_samples,
    )
    new_x = candidates.detach()
    return new_x

# %%
def bayes_opt_under_uncertainty(n_iter=100, n_batches=1, batch_size=1, delta_range=2.0):
    train_x = draw_sobol_samples(bounds=torch.tensor([[-5.0, 10.0], [0.0, 15.0]]), n=10, q=2)
    train_y = branin(train_x)

    delta = delta_range * (torch.rand(train_x.size(0), 2) - 0.5)
    train_y_noise = branin_noise(train_x, delta)

    mll, model = initialize_model(train_x, train_y_noise)

    best_obj = train_y_noise.min().item()
    best_x = train_x[train_y_noise.argmin()].detach().clone()

    # use Bayes risk as a loss function. Since its a linear operator, we can describe it by another GP
    

    for iter in range(n_iter):
        fit_gpytorch_model(mll)

        q_ei = qExpectedImprovement(model, best_obj, None)
        new_x = optimize_acq_func(q_ei, bounds=torch.tensor([[0.0, 25.0], [0.0, 15.0]]), n_restarts=10)

        delta = delta_range * (torch.rand(new_x.size(0), 2) - 0.5)
        new_y = branin_noise(new_x, delta)

        train_x = torch.cat([train_x, new_x])
        train_y_noise = torch.cat([train_y_noise, new_y])

        # mll, model = initialize_model(train_x, train_y_noise, model.state_dict())

        # use fantasy model instead?
        # updated_model = model.get_fantasy_model(new_x, new_y)

        # update posterior of the Bayes risk.
        # mu_pos_bayes = integrate(mu_posterior(f)p(delta)ddelta)
        # sigma_pos_bayes = integrate integrate(sigma_posterior(f)(x, xhat)p(delta, deltahat)ddeltaddeltahat)
        # initialize Bayes risk GP model with x and z, where x is train_x and z is integrated k prior  over the delta domain.


        # do analytical integration
        # mu_pos_bayes = 

        best_idx = train_y_noise.argmin()
        best_obj = train_y_noise[best_idx].item()
        best_x = train_x[best_idx].detach().clone()

        print(f"Iteration: {iter + 1}, Best Objective: {best_obj}, Best X: {best_x}")

    return best_x, best_obj

best_x, best_obj = bayes_opt_under_uncertainty()
print(f"Optimal Solution: X = {best_x}, f(X) = {best_obj}")
# %%
