module ADMM

# import for ADMM
using MPI # message passing interface for distributed systems
using LinearAlgebra # linear algebra
using Optim # for optimization
# using Mooncake # automatic differentiation
# using DifferentiationInterface # interface for Mooncake.jl
using Printf # fancy printing

# import for topopt
using NearestNeighbors
using NLopt
using StaticArrays, SparseArrays
using ..FEM

# export
export ADMMParams, ADMMState # structs.jl
export init, run_admm! # api.jl
export setup!, evaluate_objective, evaluate_global_regularizer, _admm_rho_changed!, objective_value # problem_virtual.jl, users must implement
# traits
export DistributionTrait, ProximalTrait # traits.jl
export DistributionMode, MPIConsensus, Serial # distribution (traits.jl)
export ProximalMode, ClosedFormProx, NumericalProx # proximal (traits.jl)
# customization hooks
export _x_update!, _z_update!, _apply_proximal! # updates.jl, unlikely that user will need to change

include("core/traits.jl")
include("core/structs.jl")
include("core/problem_virtual.jl")
include("core/updates.jl")
include("core/convergence.jl")
include("core/api.jl")

# problem implementations
include("problems/lasso.jl")
include("problems/topopt_continuous/topopt_continuous.jl")

end # of module