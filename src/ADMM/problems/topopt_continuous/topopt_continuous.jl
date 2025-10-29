# file for topopt continuous
# includes all the problem definitions and direct overwrites for ADMM.jl

export TopOptProblem

"""
Define problem
"""

Base.@kwdef mutable struct TopOptProblem{D <: ADMM.DistributionMode, T <: AbstractFloat}
    
    # problem solving space
    mesh::FEM.Mesh
    element_mask::Union{BitMatrix, Nothing} = nothing
    material::FEM.Material
    analysis_type::Union{FEM.PlaneStress, FEM.PlaneStrain, FEM.ThreeDimensional} = FEM.PlaneStress()

    # set up the problem itself
    forces::Dict{Int, <:SVector} # from node id to force vector
    boundary_dofs::Vector{Int}

    # constraints (might change for different formulation? e.g. minimizing volume instead of compliance)
    vol_frac::T # how much volume is allowed?
    σ_lim::T # what is the maximum stress?

    # projection params (ϕ → ρ)
    r_filter::T # stencil radius
    β_heaviside::T # for heaviside filter (to render sharper images)
    threshold_heaviside::T # threshold for heaviside
    β_heaviside_growth::T = 2.0 # multiplicative factor when sharpening projection
    β_heaviside_max::T = 64.0 # cap for heaviside continuation
    β_update_frequency::Int = 50 # iterations between forced β increase
    grey_band_lo::T = 0.3 # lower bound for "grey" densities
    grey_band_hi::T = 0.7 # upper bound for "grey" densities
    grey_fraction_trigger::T = 0.2 # trigger β growth if grey fraction above this
    
    # heaviside schedule options (monotonous β growth)
    use_heaviside_schedule::Bool = false # enable monotonous heaviside schedule
    heaviside_schedule_type::Symbol = :exponential # :exponential, :linear, :step
    heaviside_schedule_start::T = 1.0 # starting heaviside value
    heaviside_schedule_end::T = 16.0 # ending heaviside value
    heaviside_schedule_frequency::Int = 5 # iterations between heaviside updates
    heaviside_schedule_growth::T = 1.05 # growth factor for exponential schedule

    # numerical stability
    q_relax::T = 0.5 # avoid division by zero for near-void/void elements
    ρ_min::T = 1e-9 # set bound for void
    ρ_max::T = 1.0 # set bound for solid
    ρ_simp::T = 3.0 # default value in the literature (power law to push to 0 or 1)

    # MMA
    max_iter_mma::Int = 50 # maximum iterations
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
    α_prev::Vector{T} # previous α for convergence tracking
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
    volumes::Vector{T}
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
get the utils
"""

include("utils/_utils.jl")

"""
Custom functions
"""

function ADMM.setup!(state::ADMM.ADMMState{TopOptProblem{D,T}, Nothing}) where {D,T}

    problem = state.problem
    mesh = problem.mesh
    
    nel = length(mesh.elements)
    nnodes = length(mesh.nodes)

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

    force_dict_dof = Dict{Int, T}()

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
    volumes = FEM.get_mesh_volumes(mesh)

    # apply the element mask to ϕ
    
    ϕ = fill(T(problem.vol_frac), nel) # initialize at volume fraction
    apply_element_mask!(ϕ, problem.element_mask)

    # put together the context
    
    ctx = TopOptContext(
        nel = nel,
        ϕ = ϕ,
        ρ = zeros(T, nel),
        α = zeros(T, nel), # will be set below
        α_prev = zeros(T, nel), # for tracking convergence
        λ = ones(T, nel), # initialize λ = 1
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
        volumes = volumes,
        # σ_buffer = zeros(T, stress_size, 1)
    )

    # compute initial stresses to set α = σ̃(ϕ₀)
    tmp_state = ADMM.ADMMState(
        problem, 0, state.comm, state.rank, state.nprocs,
        0, nel, copy(ctx.ϕ), zeros(T, nel), zeros(T, nel),
        zeros(T, nel), zeros(T, nel), zeros(T, nel),
        ctx, state.params
    )
    apply_density_filter!(tmp_state)
    ctx.K = FEM.assemble_stiffness_matrix(
        problem.mesh, problem.material, problem.analysis_type;
        ρ = ctx.ρ, ρ_simp = problem.ρ_simp,
        E_min = problem.ρ_min * problem.material.E
    )
    ctx.U = FEM.solve_fem(ctx.K, ctx.f, problem.boundary_dofs)
    compute_element_stresses!(tmp_state)
    ctx.α .= ctx.σ̃
    ctx.α_prev .= ctx.α  # initialize previous α

    # ADMM state x, u, z are unused - we use ctx.ϕ, ctx.λ, ctx.α instead
    new_state = ADMM.ADMMState(
        problem, 0, state.comm, state.rank, state.nprocs,
        0, nel,  # m=0 (not used for this problem), n=nel
        T[], T[], T[],  # x, u, z unused
        T[], T[], T[],  # z_prev, primal_res, z_work unused
        ctx, state.params
    )
    
    new_state.params.μ = T(0.5) # start with smaller μ

    return new_state

end

function ADMM._x_update!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}
    # using MMA
    # pseudocode lines 5-12

    problem = state.problem
    ctx = state.ctx
    μ = state.params.μ
    nel = ctx.nel

    optimizer = NLopt.Opt(:LD_MMA, nel) # initialize mma optimizer; also try :LD_CCSAQ for conservative updates
    
    # settings
    NLopt.lower_bounds!(optimizer, zeros(T, nel))
    NLopt.upper_bounds!(optimizer, ones(T, nel))
    NLopt.xtol_rel!(optimizer, problem.mma_tol)
    NLopt.maxeval!(optimizer, problem.max_iter_mma)
    
    # Move limiter to prevent design wandering (Zhai approach)
    move_limit = T(0.1)  # Limit design variable changes to ±0.1
    NLopt.xtol_abs!(optimizer, move_limit)
    
    # Volume constraint: Σ(ρ_i * V_i) / Σ(V_i) ≤ vol_frac
    function volume_constraint(ϕ::Vector, grad::Vector)
        apply_element_mask!(ϕ, problem.element_mask)
        
        s = (ctx.H * ϕ) ./ ctx.Hs
        
        β = problem.β_heaviside
        heaviside = problem.threshold_heaviside
        denominator = tanh(β*heaviside) + tanh(β*(1 - heaviside))
        ρ_phys = (@. (tanh(β*heaviside) + tanh(β*(s - heaviside))) / denominator)
        
        active_volumes = ctx.volumes[unmasked_elements(ϕ, problem.element_mask)]
        total_volume = sum(active_volumes)
        current_volume_frac = dot(ρ_phys[unmasked_elements(ϕ, problem.element_mask)], active_volumes) / total_volume
        
        if length(grad) > 0
            dproj_ds = @. (β * (1 - tanh(β*(s - heaviside))^2)) / denominator
            # Apply mask consistently: only active elements contribute to gradient
            active_mask = unmasked_elements(ϕ, problem.element_mask)
            grad .= 0.0  # Initialize to zero
            if !isempty(active_mask)
                active_volumes = ctx.volumes[active_mask]
                active_dproj_ds = dproj_ds[active_mask]
                active_Hs = ctx.Hs[active_mask]
                grad[active_mask] .= (ctx.H[active_mask, active_mask]' * 
                                    ((active_dproj_ds .* active_volumes) ./ active_Hs)) / total_volume
            end
        end
        
        return current_volume_frac - problem.vol_frac
    end
    
    NLopt.inequality_constraint!(optimizer, volume_constraint, 1e-6)
    
    function augmented_lagrangian(ϕ::Vector, grad::Vector)
        apply_element_mask!(ϕ, problem.element_mask)

        ctx.ϕ .= ϕ # for convergence check

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

        compliance = dot(ctx.f, ctx.U)
        residual = ctx.σ̃ .- ctx.α
        augmented_term = dot(ctx.λ, residual) + (μ/2) * dot(residual, residual)

        objective = compliance + augmented_term
        
        # compute gradients ∇L using adjoint method ∂L/∂ϕ 
        # if we're lazy we could probably... get mooncake to do this for us? anyway it's now implemented explicitly
        if length(grad) > 0
            compute_gradients_adjoint!(state)
            grad .= ctx.∇L_ϕ # update NLopt gradient
        end
    
        return objective

    end

    NLopt.min_objective!(optimizer, augmented_lagrangian)
    opt_val, opt_phi, ret = NLopt.optimize(optimizer, ctx.ϕ) # min objective value, ϕ★, return code

    # Additional move limiting: clip changes to prevent design wandering
    move_limit = T(0.1)
    ϕ_prev = copy(ctx.ϕ)
    Δϕ = opt_phi .- ϕ_prev
    Δϕ_clipped = clamp.(Δϕ, -move_limit, move_limit)
    opt_phi = ϕ_prev .+ Δϕ_clipped
    
    # Ensure bounds are respected
    opt_phi = clamp.(opt_phi, T(0), T(1))

    ctx.ϕ .= opt_phi # update ctx.ϕ
    
    return nothing

end

function ADMM._z_update!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}
    # update auxiliary variable α
    # pseudocode lines 13-15

    # recompute it just in case mma picked a final state that's inconsistent with the last update
    problem = state.problem
    ctx = state.ctx
    μ = state.params.μ

    apply_element_mask!(ctx.ϕ, problem.element_mask)
    apply_density_filter!(state)
   
    ctx.K = FEM.assemble_stiffness_matrix(
            problem.mesh, 
            problem.material, 
            problem.analysis_type;
            ρ = ctx.ρ, 
            ρ_simp = problem.ρ_simp,
            E_min = problem.ρ_min * problem.material.E
        )

    ctx.U = FEM.solve_fem(ctx.K, ctx.f, problem.boundary_dofs)

    compute_element_stresses!(state)

    # actual update
    # α = clamp(σ̃ + λ/μ, 0, σ_lim) with under-relaxation to avoid clamp-lock
    ω = T(0.5)
    @. ctx.α = (1-ω)*ctx.α + ω * clamp(ctx.σ̃ + ctx.λ / μ, 0, problem.σ_lim)
    
    return nothing

end

function ADMM.check_convergence!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}
    # check convergence for topopt
    # pseudocode lines 16, 23-26

    problem = state.problem
    ctx = state.ctx
    params = state.params
    μ = params.μ

    # admm residuals
    # 1. primal
    primal_res = ctx.σ̃ .- ctx.α # r = σ̃ - α
    r_primal_norm = norm(primal_res) # should approach 0
    # 2. dual 
    r_dual_norm = μ * norm(ctx.α .- ctx.α_prev) # s = μ(α - α_prev)
    # 3. tolerances
    n = ctx.nel
    ε_abs = state.params.abstol
    ε_rel = state.params.reltol

    ϵ_primal = sqrt(n)*ε_abs + ε_rel * max(norm(ctx.σ̃), norm(ctx.α))
    ϵ_dual = sqrt(n)*ε_abs + ε_rel * μ * norm(ctx.α)

    # update λ (dual variables for stress constraints)
    @. ctx.λ = ctx.λ + μ * (ctx.σ̃ - ctx.α) # equation 12: λ = λ + μ(σ̃ - α)

    # topopt convergence
    Δ = maximum(abs.(ctx.α .- ctx.α_prev)) # pseudocode line 23: Δ = max(|α^[i] - α^[i-1]|)
    Γ = dot(ctx.ρ, ctx.volumes) / sum(ctx.volumes) # pseudocode line 24
    σ_max = maximum(ctx.σ̃) # pseudocode line 25

    # final convergence check
    admm_converged = (r_primal_norm <= ϵ_primal) && (r_dual_norm <= ϵ_dual)
    volume_satisfied = abs(Γ - problem.vol_frac) < 0.01
    stress_satisfied = σ_max <= problem.σ_lim * 1.01

    topopt_converged = admm_converged && volume_satisfied && stress_satisfied

    return r_primal_norm, r_dual_norm, ϵ_primal, ϵ_dual, topopt_converged

end

# custom run_admm! because we need to have our special convergence checks that aren't
# compatible with the generic version...
function ADMM._step!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    state.iter += 1
    ctx = state.ctx

    # save previous α for convergence tracking
    copyto!(ctx.α_prev, ctx.α)

    # update x (ϕ via MMA)
    ADMM._x_update!(state)

    # update z (α via projection)
    ADMM._z_update!(state)

    # check convergence
    residual_primal, residual_dual, epsilon_primal, epsilon_dual, topopt_converged = ADMM.check_convergence!(state)

    # Adaptive β update that reacts to grey regions and schedule
    update_heaviside_sharpness!(state)

    # Adaptive μ adjustment based on primal-dual residual balance
    adapt_μ_topopt!(state, residual_primal, residual_dual)

    return topopt_converged, residual_primal, residual_dual, epsilon_primal, epsilon_dual

end
