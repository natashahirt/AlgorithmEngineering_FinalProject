# Global Consensus Problem with Regularization

## Problem statement

From Boyd et al. 2010

The optimization problem is:

minimize ∑(i=1 to N) f_i(x_i) + g(z)
subject to x_i - z = 0

where:

- f_i is the ith objective function term (one of N terms)
- g is the global regularizer i.e. smooth constraint

The global regularizer g(z) is a function that:

1. Acts on the consensus variable z that all local variables x_i must agree on
2. Adds a penalty term to the overall objective to enforce certain properties that scales with "incorrectness"
3. Common examples include:
   - L1 regularization (g(z) = λ||z||₁) to promote sparsity
   - L2 regularization (g(z) = λ||z||₂²) to prevent overfitting
   - Total variation (g(z) = λ||∇z||₁) to enforce smoothness

Parameter λ controls the strength of regularization.

## Process

Start with local implementation (individual f + full function updates) then scale out to distributed.

## Distributed implementation

Each local subsystem `i` has `x_i` and `u_i` variables.

Two options:

1. "Local perspective": subsystem performing local processing and communicating with central coordinator for global `z` updates.
2. "Global perspective": central collector coordinating the work of a set of subsystems

My inclination is to take the local perspective and let individual nodes take agency for their work. Will also allow asynchronous updates(?)
