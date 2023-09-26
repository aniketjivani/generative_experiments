Code for playing around and testing out ideas in generative models like normalizing flows, neural ODEs and other physics + ML stuff

Potentially with a healthy dose of UQ, coming soon!

Helpful terminology:

NFE for ODE solver: See https://github.com/DiffEqML/torchdyn/issues/131

Basically, number of times the vector field has been called within `odeint`. Here we count NFE for forward and backward passes through the model, so the total NFE is 2x this number - need to double check if its really 2x! (the method for counting these separately is based on https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py)


Regularization functions definition: Use https://github.com/rtqichen/ffjord/blob/master/lib/layers/wrappers/cnf_regularization.py to get initial state of the regularization quantities and pass the augmented state to the CNF for integration. 