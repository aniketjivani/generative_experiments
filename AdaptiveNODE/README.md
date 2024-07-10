### Implementation of an adaptive Neural ODE:

Key steps:

1. True data generating equations - generate snapshots

2. Train a NODE / suitable data-driven method  in original or projected space

3. Adaptive design - target information gain in parameters and run more HF simulations

4. Adaptive design stage 2 - target predictive performance of NODE (integrated over domain) and propose designs for new HF simulations accordingly.


Use:
1. MORE Poster

2. Surrogate Bayes training

Literature:

1. Bayesian learning of dynamical systems

2. Reduced order model formulations - operator inference, structure preserving methods

3. Projection-based ROMs - how to target adaptive design for these


Setups:

1. Projection, and dynamics surrogate are both simple models

2. One or more of these are Neural Networks

3. Constrained projection / integration e.g. Hamiltonian-based
