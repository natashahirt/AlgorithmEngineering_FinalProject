# traits for easier multiple dispatch

"""
Distribution modes
- MPIConsensus: distributed
- Serial: single core fallback
"""
abstract type DistributionMode end
struct MPIConsensus <: DistributionMode end
struct Serial       <: DistributionMode end

DistributionTrait(::Type) = MPIConsensus()

"""
Proximal operator
- ClosedFormProx: there's a closed form solution that the user enters
- NumericalProx: we use numerical optimization (possibly using autograd?)
"""
abstract type ProximalMode end
struct ClosedFormProx <: ProximalMode end # has closed-form solution (e.g., LASSO soft-threshold)
struct NumericalProx <: ProximalMode end # needs numerical optimization (e.g. Optim.jl)

# Default: no global regularizer (identity prox)
ProximalTrait(::Type) = ClosedFormProx()