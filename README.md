Code for playing around and testing out ideas in generative models like normalizing flows, neural ODEs and other physics + ML stuff

Potentially with a healthy dose of UQ, coming soon!

Done as a first try:
Continuous Normalizing Flow with Regularized NODE
Score-based diffusion model for toy problem.

Folders:
EDL - trying toy examples from [A.Amini - Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning/tree/main/evidential_deep_learning)

Near-future Exploration:
1. Time reversible ODEs - compare with odd functions (x^3) and others (non-reversible) - x^2 + 1. Re-implement in PyTorch since original code is in Keras

2. Incorporate Hamiltonian prior

3. Check longer time rollouts on all three methods in potentially 1D case (fewer complexities in setting up training)



Helpful terminology:

NFE for ODE solver: See https://github.com/DiffEqML/torchdyn/issues/131

Basically, number of times the vector field has been called within `odeint`. Here we count NFE for forward and backward passes through the model, so the total NFE is 2x this number - need to double check if its really 2x! (the method for counting these separately is based on https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py)


Regularization functions definition: Use https://github.com/rtqichen/ffjord/blob/master/lib/layers/wrappers/cnf_regularization.py to get initial state of the regularization quantities and pass the augmented state to the CNF for integration.


