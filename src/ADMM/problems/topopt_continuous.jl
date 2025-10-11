export TopOptContinuous

"""
Define problem
"""

Base.@kwdef mutable struct TopOptContinuous{D <: ADMM.DistributionMode}
    A::AbstractMatrix{Float64} # design matrix (m x n)
    b::AbstractVector{Float64} # observations (m x 1)
    λ::Float64 = 0.1 # L1 penalty parameter
    distribution::D = ADMM.Serial() # default to serial
end
