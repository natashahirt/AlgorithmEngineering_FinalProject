export TopOptProblem

"""
Define problem
"""

Base.@kwdef mutable struct TopOptProblem{D <: ADMM.DistributionMode, T <: AbstractFloat}
    
    # problem solving space
    mesh::FEM.Mesh
    material::FEM.Material
    analysis_type::Union{FEM.PlaneStress, FEM.PlaneStrain, FEM.ThreeDimensional} = FEM.PlaneStress()

    # set up the problem itself
    forces::Dict{Int, SVector} # from node id to force vector
    boundary_dofs::Vector{Int}

    # constraints (might change for different formulation? e.g. minimizing volume instead of compliance)
    vol_frac::T # how much volume is allowed?
    σ_lim::T # what is the maximum stress?

    # projection params (ϕ → ρ)
    r_filter::T # stencil radius
    β_heaviside::T # for heaviside filter (to render sharper images)
    η_heaviside::T # threshold for heaviside

    # numerical stability
    q_relax::T = 0.5 # avoid division by zero for near-void/void elements
    ρ_min::T = 1e-9 # set bound for void
    ρ_max::T = 1.0 # set bound for solid
    ρ_simp::T = 3.0 # default value in the literature (power law to push to 0 or 1)

    # MMA
    max_iter_mma::Int = 100 # maximum iterations
    mma_tol::T = 1e-4 # convergence tolerance

    # distribution type
    distribution::D = ADMM.Serial()

end

# cache for performance
Base.@kwdef mutable struct TopOptContext{T <: AbstractFloat}

    nel::Int # number of elements
    ϕ::Vector{T} # raw design vars (continuous)
    ρ::Vector{T} # filtered/projected density vars (physical, binary)

    # for ADMM 
    α::Vector{T} # auxiliary stress variable per element
    λ::Vector{T} # lagrange multiplier per element

    # global FEM solution
    U::Vector{T} # displacement
    K::SparseMatrixCSC{T,Int} # stiffness matrix K(ρ)
    f::Vector{T} # force vector

    # element level stress values
    σ::Matrix{T} # full stress components for all elements (stress_size × nel)
    σ̄::Vector{T} # von Mises stress per element σ̄_e
    σ̃::Vector{T} # relaxed stress per element σ̃_e 

    # filter operator (sparse matrix for efficiency)
    H::SparseMatrixCSC{T,Int} # filter matrix ρ = H * ϕ
    Hs::Vector{T} # normalization (sum of filter weights per element)

    # gradient computation by adjoint method
    ∇L_ϕ::Vector{T} # gradient of Lagrangian with regard to ϕ
    λ_adjoint::Vector{T} # adjoint variables for displacement

    # work buffers to prevent allocations
    K_elem_buffer::Matrix{T} # stiffness matrix buffer (reused for each element)
    ϵ_buffer::Matrix{T} # strain buffer (reused for each element)
    # σ_buffer::Matrix{T} # stress buffer (reused for each element) -- add if we do things on the fly later on

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

function ADMM.setup!(state::ADMM.ADMMState{TopOptProblem{D}, Nothing}) where {D}

    problem = state.problem
    mesh = problem.mesh
    
    nel = length(mesh.elements)
    nnodes = length(mesh.nodes)
    T = eltype(mesh.nodes[1].coords) # get element type for type param

    dim = length(mesh.nodes[1].coords)  # 2 or 3 dimensions
    ndof = nnodes * dim

    # dimensionality detection and initialization

    if dim == 2
        elem_dof = 8  # 4 nodes × 2 DOF/node
        strain_size = 3  # εxx, εyy, γxy
        stress_size = 3  # σxx, σyy, τxy
        if !isdefined(problem, :analysis_type)
            problem.analysis_type = FEM.PlaneStress()
        end
    elseif dim == 3
        elem_dof = 24  # 8 nodes × 3 DOF/node
        strain_size = 6  # εxx, εyy, εzz, γxy, γxz, γyz
        stress_size = 6  # σxx, σyy, σzz, τxy, τxz, τyz
        problem.analysis_type = FEM.ThreeDimensional()
    else
        error("Unsupported dimension: $dim")
    end

    # build the filter matrix

    H, Hs = build_filter_matrix(problem)

    # assemble the force vector
    # force vector is input as a dictionary of nodes to forces but FEM.assemble_force_vector needs from dof to forces

    force_dict_dof = Dict{Int, Float64}()

    for (node_id, force_vec) in problem.forces
        node_dofs = FEM.get_node_dofs(mesh, node_id)
        for (local_idx, global_dof) in enumerate(node_dofs)
            force_component = force_vec[local_idx]
            if haskey(force_dict_dof, global_dof)
                force_dict_dof[global_dof] += force_component
            else
                force_dict_dof[global_dof] = force_component
            end
        end
    end

    f_global = FEM.assemble_force_vector(force_dict_dof, ndof)

    # put together the context

    ctx = TopOptContext(
        nel = nel,
        ϕ = ones(T, nel) * 0.5, # initial ϕ
        ρ = zeros(T, nel),
        α = zeros(T, nel),
        λ = zeros(T, nel),
        U = zeros(T, ndof),
        K = spzeros(T, ndof, ndof),
        f = f_global,
        σ = zeros(T, stress_size, nel), # stress_size × nel matrix
        σ̄ = zeros(T, nel),
        σ̃ = zeros(T, nel),
        H = H,
        Hs = Hs,
        ∇L_ϕ = zeros(T, nel),
        λ_adjoint = zeros(T, ndof),
        K_elem_buffer = zeros(T, elem_dof, elem_dof),
        ϵ_buffer = zeros(T, strain_size, 1),
        # σ_buffer = zeros(T, stress_size, 1)
    )

    return ADMM.ADMMState(
        problem, state.comm, state.rank, state.nprocs,
        0, nel,  # m=0 (not used for this problem), n=nel
        zeros(T, nel), zeros(T, nel), zeros(T, nel),  # x, u, z
        zeros(T, nel), zeros(T, nel), zeros(T, nel),  # z_prev, primal_res, z_work
        ctx, state.params
    )

end

function mma_loop(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    problem = state.problem
    ctx = state.ctx

    # ϕ → ρ: apply density filter and Heaviside projection (in-place update)
    apply_density_filter!(state)
    
    # build stiffness matrix K(ρ) with SIMP interpolation
    ctx.K = FEM.assemble_stiffness_matrix(
        problem.mesh, 
        problem.material, 
        problem.analysis_type;
        ρ = ctx.ρ, 
        ρ_simp = problem.ρ_simp,
        E_min = problem.ρ_min * problem.material.E
    )
    
    # solve FEM: K(ρ)U = f with boundary conditions
    ctx.U = FEM.solve_fem(ctx.K, ctx.f, problem.boundary_dofs)
    
    # Compute element stresses σ̄ (von Mises) and σ̃ (relaxed)
    compute_element_stresses!(state)
    
    # compute gradients ∇L using adjoint method ∂L/∂ϕ 
    compute_gradients_adjoint!(state)

    
end