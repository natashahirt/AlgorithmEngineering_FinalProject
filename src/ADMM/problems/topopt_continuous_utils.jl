# utility functions for topopt_continuous.jl

function build_filter_matrix(problem::TopOptProblem{D,T}) where {D,T}

    mesh = problem.mesh
    r_filter = problem.r_filter
    nel = length(mesh.elements)

    centroids = FEM.get_mesh_centroids(mesh) # dim × nel
    volumes = FEM.get_mesh_volumes(mesh) # 1 × nel

    # nearest neighbors is a little elaborate when we have a pre-set grid
    # but useful if there's a more varied case
    tree = KDTree(centroids) # built in nearest neighbors
    
    # find the neighbors that lie within r_filter
    source_idx = Int[]
    neighbor_idx = Int[]
    weights = T[]

    for e in 1:nel

        neighbors = inrange(tree, centroids[e], r_filter)

        for i in neighbors

            dist = norm(centroids[e] - centroids[i])
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

function heaviside_projection(problem::TopOptProblem{D,T}, ϕ::T) where {D,T <: AbstractFloat}

    β = problem.β_heaviside # sharpness (larger = more binary)
    η = problem.η_heaviside # threshold (generally 0.5)

    tanh_βη = tanh(β * η)

    ρ = (tanh_βη + tanh(β * (ϕ - η))) / (tanh_βη + tanh(β * (one(T) - η)))

    return ρ

end

function apply_density_filter!(problem::TopOptProblem{D,T}) where {D,T}

    ctx = problem.ctx

    mul!(ctx.ρ, ctx.H, ctx.ϕ) # ctx.ρ = H * ϕ
    ctx.ρ ./= ctx.Hs # element-wise division

    @inbounds for i in eachindex(ctx.ρ)
        ctx.ρ[i] = heaviside_projection(problem, ctx.ρ[i])
    end

    clamp!(ctx.ρ, problem.ρ_min, problem.ρ_max)

end

function q_relaxation(q_relax::T, ρ::Vector{T}, σ::Vector{T}) where {T}
    # from Le et al. 2010
    σ̄ = FEM.compute_von_mises_stress(σ)
    σ̃ = ρ.^q_relax .* σ̄
    return σ̄, σ̃
end