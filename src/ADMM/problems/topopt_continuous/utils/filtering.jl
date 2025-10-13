# utility functions -- filtering

"""
setup -- sparse filter matrix H and normalization vector Hs
KDTree for efficient neighbor search within filter radius (replace with stamping for more regular problems?)
"""
function build_filter_matrix(problem::TopOptProblem{D,T}) where {D,T}

    mesh = problem.mesh
    r_filter = problem.r_filter
    nel = length(mesh.elements)
    T_float = eltype(mesh.nodes[1].coords)

    centroids = FEM.get_mesh_centroids(mesh) # dim × nel
    volumes = FEM.get_mesh_volumes(mesh) # 1 × nel

    # nearest neighbors is a little elaborate when we have a pre-set grid
    # but useful if there's a more varied case
    tree = KDTree(centroids) # built in nearest neighbors
    
    # find the neighbors that lie within r_filter
    source_idx = Int[]
    neighbor_idx = Int[]
    weights = T_float[]

    for e in 1:nel

        neighbors = inrange(tree, centroids[:, e], r_filter)

        for i in neighbors

            dist = norm(centroids[:, e] - centroids[:, i])
            weight = (r_filter - dist) * volumes[i]

            push!(source_idx, e)
            push!(neighbor_idx, i)
            push!(weights, weight)

        end

    end

    H = sparse(source_idx, neighbor_idx, weights, nel, nel)
    Hs = vec(sum(H, dims=2))

    return H, Hs

end

"""
heaviside projection to improve sharpness
"""
function heaviside_projection(problem::TopOptProblem{D,T}, ϕ::T) where {D,T <: AbstractFloat}

    β = problem.β_heaviside # sharpness (larger = more binary)
    η = problem.η_heaviside # threshold (generally 0.5)

    tanh_βη = tanh(β * η)

    ρ = (tanh_βη + tanh(β * (ϕ - η))) / (tanh_βη + tanh(β * (one(T) - η)))

    return ρ

end

"""
apply density filter and Heaviside projection: ϕ → ρ (in place)
"""
function apply_density_filter!(state::ADMM.ADMMState{TopOptProblem{D,T}, TopOptContext{T}}) where {D,T}

    problem = state.problem
    ctx = state.ctx

    # linear filter (spatial smoothing)
    mul!(ctx.ρ, ctx.H, ctx.ϕ) # ctx.ρ = H * ϕ
    ctx.ρ ./= ctx.Hs # element-wise division

    # heaviside projection (thresholding)
    @inbounds for i in eachindex(ctx.ρ)
        ctx.ρ[i] = heaviside_projection(problem, ctx.ρ[i])
    end

    # clamp to bounds
    clamp!(ctx.ρ, problem.ρ_min, problem.ρ_max)

    return nothing
end
