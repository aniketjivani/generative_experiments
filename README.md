Code for playing around and testing out ideas in generative models like normalizing flows, neural ODEs and other (physics + ML) stuff

Potentially with a healthy dose of UQ, coming soon! 

The ideal combination we wish to investigate is:
1. Building architectures that handle small data well (incorporate better inductive bias) - this is critical for scientific domains where large-scale studies are prohibitive and it may be necessary to work with legacy data.

2. UQ methods that play well with deep learning - advances in GPs + Bayesian and non-Bayesian methods that have potential to scale, provide well-calibrated measures of uncertainty (without the trappings of conformal prediction where coverage guarantees have slack in small calibration data settings).

3. Design experiments for surrogate model parameters e.g. NN weights through Bayesian OED formulations.

Folders:

1. EDL - trying toy examples from [A.Amini - Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning/tree/main/evidential_deep_learning)

2. pinn_curriculum - trying examples from [Krishnapriyan et al.](https://github.com/a1k12/characterizing-pinns-failure-modes/blob/main/pbc_examples/main_pbc.py) for characterizing PINN failure modes - their paper can be found at: 

Near-future Exploration:
1. Time reversible ODEs. Re-implement in PyTorch since original code doesn't play nice with TF2.0 - https://github.com/inhuh/trs-oden

2. Graph NODES

3. Check longer time rollouts in time-reversible settings and with graph-based methods.



**Helpful terminology:**

NFE for ODE solver: See https://github.com/DiffEqML/torchdyn/issues/131

Basically, number of times the vector field has been called within `odeint`. Here we count NFE for forward and backward passes through the model, so the total NFE is 2x this number - need to double check if its really 2x! (the method for counting these separately is based on https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py)


Regularization functions definition: Use https://github.com/rtqichen/ffjord/blob/master/lib/layers/wrappers/cnf_regularization.py to get initial state of the regularization quantities and pass the augmented state to the CNF for integration.


(credits to all the awesome researchers who make their code freely available)

