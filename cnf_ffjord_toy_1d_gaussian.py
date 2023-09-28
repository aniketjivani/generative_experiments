# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Repeat the same example as the earlier notebook, but use CNF construction with regularization, add NFE property too.

# %%
import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Union, Callable
from torch.autograd import grad

import seaborn as sns

import six
import math

# %%
plt.rc("axes.spines", right=True, top=True)
plt.rc("figure", dpi=300, 
       figsize=(9, 3)
      )
plt.rc("font", family="serif")
plt.rc("legend", edgecolor="none", frameon=True)

# %%
device = torch.device('cuda:' + '0' if torch.cuda.is_available() else 'cpu')

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--adjoint", action="store_false")

parser.add_argument('--l2int', 
                    type=float, 
                    default=0.1, 
                    help="int_t ||f||_2")
parser.add_argument('--JFrobint', 
                    type=float, 
                    default=0.1, 
                    help="int_t ||df/dx||_F")
parser.add_argument('--num_samples', type=int, default=1000, help="number of samples")

parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--niters', type=int, default=1200, help="number of iterations")
parser.add_argument('--use_reg', type=bool, default=True)

args = parser.parse_args(args=())

# %%
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import reg_lib
import utils

# %%
REGULARIZATION_FNS = {
    "l1int": reg_lib.l1_regularzation_fn,
    "l2int": reg_lib.l2_regularzation_fn,
    "dl2int": reg_lib.directional_l2_regularization_fn,
    "JFrobint": reg_lib.jacobian_frobenius_regularization_fn,
    "JdiagFrobint": reg_lib.jacobian_diag_frobenius_regularization_fn,
    "JoffdiagFrobint": reg_lib.jacobian_offdiag_frobenius_regularization_fn,
}

INV_REGULARIZATION_FNS = {v: k for k, v in six.iteritems(REGULARIZATION_FNS)}


# %%
def append_regularization_to_log(log_message, regularization_fns, reg_states):
    for i, reg_fn in enumerate(regularization_fns):
        log_message = log_message + " | " + INV_REGULARIZATION_FNS[reg_fn] + ": {:.8f}".format(reg_states[i].item())
    return log_message


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


# %%
regularization_fns, regularization_coeffs = create_regularization_fns(args)

regularization_fns, regularization_coeffs


# %%
def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


# %%
noise_samples = torch.distributions.Normal(0, 1).sample((1000, 1))
noise_samples.shape


# %%
def get_batch(num_samples):
     # Define the mean of each gaussian
    means = np.array([-3.5, 0.0, 3.5])
    # Define the weights for each gaussian
    weights = np.array([0.2, 0.2, 0.6])
    
    scales = np.array([1.5, 0.6, 0.6])
    
    weights /= np.sum(weights)
    # randomly choose a gaussian for each sample
    components = np.random.choice(means.size, size=num_samples, p=weights)
    # sample from the chosen gaussians
    points = np.random.normal(means[components], scales[components], size=num_samples)
    return torch.from_numpy(points).float().view(-1, 1).to(device), torch.zeros(num_samples, 1).type(torch.float32).to(device)


# %%
class ODEfunc(nn.Module):
    """
    Create an ODEfunc class where we take in dimension of input z as an argument and create a neural network with one hidden layer, and non-linear activation tanh between each linear layer. The output of the neural network is the derivative of z with respect to t, dz/dt.
    """
    def __init__(self, dim, hidden_dim, nonlinear=nn.ELU(inplace=True)):
        super(ODEfunc, self).__init__()
        self.nl = nonlinear
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #         self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.nl(out)
        out = self.fc2(out)
        out = self.nl(out)
        #         out = self.fc3(out)
        #         out = self.nl(out)
        out = self.fc4(out)
        return out


# %%
class CNF_plain(nn.Module):
    """
    CNF without any regularization on the kinetic energy or the Jacobian. Trace estimator specifies whether to calculate exact trace or use Hutchinson's trace estimator.
    """
    def __init__(self, 
                 net:nn.Module, 
                 trace_estimator:Union[Callable, None]=None, 
                 noise_dist=None,
                #  integration_time=torch.Tensor([1, 0]).float()
                 ):
        super(CNF_plain, self).__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else trace_df_dz
        self.noise_dist = noise_dist

    @property
    def nfe(self):
        return self.net.nfe

    @nfe.setter
    def nfe(self, value):
        self.net.nfe = value


    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.net(t, z).view(batchsize, 1)
            
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


# %%
class CNF_regularized(nn.Module):
    """
    CNF with regularization on the kinetic energy or the Jacobian. for now we still use exact trace.
    """
    def __init__(self, 
                 net:nn.Module, 
                 trace_estimator:Union[Callable, None]=None, 
                 noise_dist=None,
                 ):
        super(CNF_regularized, self).__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else trace_df_dz
        self.noise_dist = noise_dist
        self.regularization_fns = regularization_fns

    @property
    def nfe(self):
        return self.net.nfe

    @nfe.setter
    def nfe(self, value):
        self.net.nfe = value

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz_dt = self.net(t, z).view(batchsize, 1)
            
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

            reg_states = tuple(reg_fn(z, logp_z, dz_dt, dlogp_z_dt) for reg_fn in self.regularization_fns)

        return (dz_dt, dlogp_z_dt) + reg_states


# %%
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# %% [markdown]
# We will test this on a 1D Gaussian distribution, see if the NFE works as expected. Then we will include regularization in an additional class.

# %%
my_net = ODEfunc(1, 32).to(device)
my_net

# %%
my_cnf_plain = CNF_plain(my_net).to(device)
my_cnf_reg = CNF_regularized(my_net).to(device)

# %%
for name, param in my_cnf_plain.named_parameters():
    print(name, param.shape)

# %%
if args.use_reg:
    optimizer = optim.Adam(my_cnf_reg.parameters(), lr=1e-3)
else:
    optimizer = optim.Adam(my_cnf_plain.parameters(), lr=1e-3)

# %%
p_z0 = torch.distributions.Normal(
    loc=torch.tensor(0.0).to(device),
    scale=torch.tensor(3.0).to(device)
        )

# %%
# x1, logp_diff_t1 = get_batch(args.num_samples)
# tt = 0.5
# with torch.set_grad_enabled(True):
#     x1.requires_grad_(True)
#     dz_dt = my_net(tt, x1).view(1000, 1)
#     dlogp_z_dt = -trace_df_dz(dz_dt, x1).view(1000, 1)

# # logp_diff_t1 is all zeros as it should be.
# # dz_dt.shape, dlogp_z_dt.shape # (1000, 1), (1000, 1)

# reg1 = reg_lib.l2_regularzation_fn(x1, logp_diff_t1, dz_dt, dlogp_z_dt)
# reg1

# reg2 = reg_lib.jacobian_frobenius_regularization_fn(x1, logp_diff_t1, dz_dt, dlogp_z_dt)
# reg2

# reg1, reg2

# all_states = odeint(my_cnf_reg, (x1, logp_diff_t1) + tuple(torch.tensor(0.) for _ in range(len(regularization_coeffs))), torch.Tensor([1, 0]).float().to(device))

# all_states[2:]

# %%
logger = utils.get_logger(logpath=os.path.join(os.getcwd(), "logs", "cnf_ffjord_toy_1d_gaussian.log"))

# %%
f_nfe_meter = RunningAverageMeter()
b_nfe_meter = RunningAverageMeter()
loss_meter = RunningAverageMeter()

for itr in range(1, args.niters + 1):
    optimizer.zero_grad()

    x, logp_diff_t1 = get_batch(args.num_samples)

    if args.use_reg:
        reg_states_init = tuple(torch.tensor(0.) for _ in range(len(regularization_coeffs)))
        all_states  = odeint(my_cnf_reg, 
                            (x, logp_diff_t1) + reg_states_init, 
                            torch.Tensor([1, 0]).float().to(device), 
                            method='dopri5', 
                            atol=1e-5, 
                            rtol=1e-5)

        z_t, logp_diff_t = all_states[0], all_states[1]
        reg_states_final = all_states[2:]

        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = p_z0.log_prob(z_t0).to(device).view(-1) - logp_diff_t0.view(-1) # evaluate change of variables

        loss = torch.mean(-logp_x)
        reg_loss = sum(reg_state[-1] * coeff for reg_state, coeff in zip(reg_states_final, regularization_coeffs) if coeff != 0)

        loss = loss + reg_loss

        nfe_forward = my_cnf_reg.nfe
        my_cnf_reg.nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = my_cnf_reg.nfe
        my_cnf_reg.nfe = 0

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        loss_meter.update(loss.item())

    else:
        all_states  = odeint(my_cnf_plain, 
                            (x, logp_diff_t1), 
                            torch.Tensor([1, 0]).float().to(device), 
                            method='dopri5', 
                            atol=1e-5, 
                            rtol=1e-5)

        z_t, logp_diff_t = all_states[0], all_states[1]

        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]

        logp_x = p_z0.log_prob(z_t0).to(device).view(-1) - logp_diff_t0.view(-1) # evaluate change of variables

        loss = torch.mean(-logp_x)
        
        nfe_forward = my_cnf_plain.nfe
        my_cnf_plain.nfe = 0

        loss.backward()
        optimizer.step()

        nfe_backward = my_cnf_plain.nfe
        my_cnf_plain.nfe = 0

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        loss_meter.update(loss.item())


    logger.info("Iter {:04d} | Loss {:.4f} | NFE-F {:.1f} | NFE-B {:.1f} |".format(
                        itr, loss_meter.avg, f_nfe_meter.avg, b_nfe_meter.avg
                    ))

# %%
batch_pred = 300
samples_pred = 1000
time_pred = 100
xfinal, logp_diff_t1_final = get_batch(samples_pred)

# %%
# it seems incorrect that the my_cnf is restricted by number of samples instead of dimensionality of input - wrong architecture!!!

z_t_final, logp_diff_t_final = odeint(
            my_cnf_plain,
            (xfinal, logp_diff_t1_final),
            torch.tensor(np.linspace(1, 0, time_pred)).type(torch.float32).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

# %%
# PLOT BASE DISTRIBUTION i.e. p_z0.sample, LEARNT DYNAMICS i.e. z_t_final and TARGET DISTRIBUTION i.e. x IN A 3x1 PLOT WITH SEABORN

target_samples = xfinal.cpu().detach().numpy()
base_samples_actual = p_z0.sample((samples_pred, 1)).cpu().detach().numpy() # actually, we should use these and integrate in reverse, but never mind.
base_samples_predicted = z_t_final.cpu().detach().numpy()[-1, :, 0]

learnt_trajectories = z_t_final.cpu().detach().numpy()[:, :, 0]

fig, ax = plt.subplots(3, 1, figsize=(5, 10), sharex=True)

axs = ax.ravel()
sns.histplot(base_samples_predicted, ax=axs[0], kde=True, label="")
axs[0].set_title("Base Samples (predicted)")

traj_idx = np.random.choice(learnt_trajectories.shape[1], size=batch_pred, replace=False)



for i in traj_idx:
    axs[1].plot(learnt_trajectories[:, i], np.linspace(1, 0, time_pred), label='')

axs[1].set_ylim((1, 0))


sns.histplot(target_samples, ax=axs[2], kde=True, label="")
axs[2].set_title("Original target samples")

fig.tight_layout()

# %%

# %%
