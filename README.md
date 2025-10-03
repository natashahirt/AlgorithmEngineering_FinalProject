# AlgorithmEngineering_FinalProject
Stress-constrained optimization in Julia using the ADMM optimizer.
Research as part of the Digital Structures Research Group, begun as a final project for MIT 6.6050 Algorithm Engineering.

The overarching goals of this project are to develop:
* A fast, Julia-native, parallelized implementation of the ADMM optimizer (if satisfactory, to be forked off as an independent package).
* An implementation of stress-constrained, vertex-centric topology optimization (elements are no longer first-class citizens but a byproduct of the vertex-centric topopt process) that uses ADMM as the backend.

## Repository Layout
* `AlgEng/Manifest.toml` and `AlgEng/Project.toml`: Julia environment manifest and dependencies for the `AlgEng` package.
* `AlgEng/src/AlgEng.jl`: Package entry point.
* `LICENSE`: Project license (MIT).
