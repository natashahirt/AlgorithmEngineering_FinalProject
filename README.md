# AlgorithmEngineering_FinalProject

Stress-constrained optimization in Julia using the ADMM optimizer.
Research as part of the Digital Structures Research Group, begun as a final project for MIT 6.6050 Algorithm Engineering.

The overarching goals of this project are to develop:

* A fast, Julia-native, parallelized implementation of the ADMM optimizer (if satisfactory, to be forked off as an independent package).
* An implementation of stress-constrained, vertex-centric topology optimization (elements are no longer first-class citizens but a byproduct of the vertex-centric topopt process) that uses ADMM as the backend.

## Repository Layout

* `Manifest.toml` and `Project.toml`: Julia environment manifest and dependencies for the ADMM package.
* `src/ADMM.jl`: Package entry point containing exports and includes.
* `src/core/`: Core ADMM implementation files:
  * `traits.jl`: Distribution and proximal operator traits
  * `structs.jl`: Parameter and state structs
  * `problem_virtual.jl`: Abstract problem interface
  * `updates.jl`: ADMM update steps
  * `convergence.jl`: Convergence criteria
  * `api.jl`: Public API
* `src/problems/`: Concrete problem implementations
  * `lasso.jl`: LASSO regression example
* `examples/`: Example usage
  * `lasso_run.jl`: LASSO regression demo
* `test/`: Test suite
* `LICENSE`: Project license (MIT).
