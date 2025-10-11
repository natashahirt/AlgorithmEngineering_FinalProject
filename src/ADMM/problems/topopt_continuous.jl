export TopOptProblem

"""
Define problem
"""

Base.@kwdef mutable struct TopOptProblem{D <: ADMM.DistributionMode}
    
    # problem solving space
    mesh::FEM.Mesh
    material::FEM.Material

    # set up the problem itself
    forces::Dict{Int, SVector} # from node id to force vector -- can definitely reconsider this format
    boundary_dof

end

mutable struct TopOptContext



end

"""
Define traits
"""

# Distribute across multiple processes
# ADMM.DistributionTrait(::Type{<:TopOptProblem}) = ADMM.MPIConsensus()
ADMM.DistributionTrait(::Type{<:TopOptProblem{D}}) where D = D()

# Closed-form proximal operator (soft-thresholding)
ADMM.ProximalTrait(::Type{<:TopOptProblem}) = ADMM.ClosedFormProx()


"""
Custom functions
"""