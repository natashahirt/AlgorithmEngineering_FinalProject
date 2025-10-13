# AlgorithmEngineering_FinalProject

Stress-constrained topology optimization in Julia using ADMM.  
Final project for MIT 6.6050 Algorithm Engineering.

## Features

* Generic ADMM framework with custom problem interface
* 2D/3D finite element method (FEM) with plane stress/strain
* Stress-constrained topology optimization using adjoint gradients
* Heaviside projection with continuation for binary designs
* MPI support for distributed optimization

## Repository Layout

* `src/`
  * `ADMM/`: ADMM optimizer framework
    * `core/`: Generic ADMM implementation (traits, updates, convergence)
    * `problems/`: Problem implementations (Lasso, topology optimization)
  * `FEM/`: Finite element analysis module
* `examples/`
  * `ADMM/`: Example scripts and outputs for ADMM solver
    * `output`: Saved images from `topopt_run` tests
    * `topopt_run.jl`: Michell truss optimization example (custom topopt implementation)
    * `lasso_run.jl`: LASSO regression demo (uses generic ADMM framework and is quite nice)
  * `FEM/`: Example scripts for 2d and 3d FEM problems
* `test/`: Test suite
