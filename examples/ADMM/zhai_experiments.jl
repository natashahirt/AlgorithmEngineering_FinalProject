"""
Minimal Topology Optimization Example - L-domain
"""

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)))

using AlgorithmEngineering.ADMM
using AlgorithmEngineering.FEM
using StaticArrays
using LinearAlgebra
using Printf
using GLMakie

println("="^70)
println("Topology Optimization: Michell Truss (ADMM)")
println("="^70)

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Geometry (aspect ratio 2:1)
nelx = 150         # Elements in x (keep small for speed)
nely = 150         # Elements in y
Lx = nelx         # Length [m]
Ly = nely         # Height [m]

# Material
E = 1.0           # Young's modulus (normalized)
ν = 0.3           # Poisson's ratio

# Optimization parameters
vol_frac = 0.30   # Target volume fraction (30% material for L-shape)
σ_lim = 5.      # Stress limit (L-shape example)
max_iter = 100   # Maximum ADMM iterations

# SIMP and filtering parameters
ρ_simp = 3.0      # SIMP penalty parameter (Zhai standard)
r_filter = 1.5    # Density filter radius (1.5× element width)

# Heaviside projection parameters
β_heaviside = 1.0         # Initial Heaviside parameter (start gentle)
threshold_heaviside = 0.5         # Heaviside threshold (midpoint of density scale)
β_heaviside_max = 64.0    # Maximum Heaviside parameter
β_update_frequency = 50   # Heaviside update frequency (slower sharpening)

# heaviside schedule parameters (monotonous β growth)
use_heaviside_schedule = true   # Enable monotonous heaviside schedule (Zhai approach)
heaviside_schedule_type = :exponential  # :exponential, :linear, :step
heaviside_schedule_start = 1.0  # Starting β value
heaviside_schedule_end = 16.0   # Ending β value  
heaviside_schedule_frequency = 5 # Update every 5 iterations
heaviside_schedule_growth = 1.05 # Growth factor for exponential schedule

# MMA parameters (for inner subproblems)
max_iter_mma = 100        # Maximum MMA iterations
mma_tol = 1e-3            # MMA convergence tolerance

# ADMM parameters
μ = 0.5                   # Penalty parameter (Zhai starting value)
reltol = 1e-2             # Relative tolerance
abstol = 1e-3             # Absolute tolerance
adaptive_μ = false        # Disable adaptive penalty (use monotone schedule)

# ============================================================================
# PROBLEM SETUP
# ============================================================================

println("\nProblem Setup:")
println("  Mesh: $(nelx)×$(nely) = $(nelx*nely) elements")
println("  Volume fraction: $(vol_frac*100)%")
println("  Stress limit: $(σ_lim)")

# ============================================================================
# MESH & BOUNDARY CONDITIONS
# ============================================================================

println("\nGenerating mesh...")
mesh = FEM.generate_rectangular_mesh(Lx, Ly, nelx, nely);
element_mask = FEM.rect_mask(mesh, (61,61), (150,150))

# Get boundary nodes
top_nodes = FEM.get_boundary_nodes(mesh, "top")
right_nodes = FEM.get_boundary_nodes(mesh, "right")

# Boundary conditions: Fix the top row of nodes (ux=0, uy=0 for all top_nodes)
boundary_dofs = Int[]
for node_id in top_nodes
    dofs = FEM.get_node_dofs(mesh, node_id)
    push!(boundary_dofs, Int(dofs[1]))  # ux (ensure integer)
    push!(boundary_dofs, Int(dofs[2]))  # uy (ensure integer)
end

println("  Boundary: pins at top left and bottom left corners")

# Loading: downward force at specific right edge nodes (x = 150, y = [91,92,93,94])

# Find nodes at x = 150, y = [60, 59, 58, 57]
selected_y = [60, 59, 58, 57]
right_force_nodes = [node.id for node in values(mesh.nodes) if isapprox(node.coords[1], 150.0; atol=1e-5) && (Int(round(node.coords[2])) in selected_y)]

forces = Dict(nid => SVector(0.0, -1.0) for nid in right_force_nodes)

println("  Load: 1N downward at right edge nodes x=150, y=$(selected_y) (node ids: $(right_force_nodes))")

# ============================================================================
# TOPOLOGY OPTIMIZATION PROBLEM
# ============================================================================

println("\nCreating topology optimization problem...")

material = FEM.LinearElastic(E, ν, 1.0)  # E, ν, density

problem = ADMM.TopOptProblem(
    mesh = mesh,
    element_mask = element_mask,
    material = material,
    analysis_type = FEM.PlaneStress(),
    forces = forces,
    boundary_dofs = boundary_dofs,
    vol_frac = vol_frac,
    σ_lim = σ_lim,
    r_filter = r_filter,
    β_heaviside = β_heaviside,
    threshold_heaviside = threshold_heaviside,
    β_heaviside_max = β_heaviside_max,
    β_update_frequency = β_update_frequency,
    ρ_simp = ρ_simp,
    max_iter_mma = max_iter_mma,
    mma_tol = mma_tol,
    # heaviside schedule options (Zhai approach)
    use_heaviside_schedule = use_heaviside_schedule,
    heaviside_schedule_type = heaviside_schedule_type,
    heaviside_schedule_start = heaviside_schedule_start,
    heaviside_schedule_end = heaviside_schedule_end,
    heaviside_schedule_frequency = heaviside_schedule_frequency,
    heaviside_schedule_growth = heaviside_schedule_growth
)

# ============================================================================
# ADMM SOLVER
# ============================================================================

println("\nInitializing ADMM solver...")

admm_params = ADMM.ADMMParams(
    μ = μ,                    # Penalty parameter
    reltol = reltol,          # Relative tolerance
    abstol = abstol,          # Absolute tolerance
    adaptive_μ = adaptive_μ   # Enable adaptive penalty
)

state = ADMM.init(problem, params=admm_params)

println("  Variables: $(state.n) elements")
println("  DOFs: $(length(state.ctx.U))")
println("  Initial μ: $(admm_params.μ)")

# ============================================================================
# OPTIMIZATION
# ============================================================================

println("\n" * "="^70)
println("Starting optimization...")
println("="^70)

state_final, iters, converged = ADMM.run_admm!(state, max_iter=max_iter, verbose=true)

println("\n" * "="^70)
println("Optimization complete!")
println("="^70)

# ============================================================================
# RESULTS
# ============================================================================

println("\nResults:")
println("  Iterations: $(iters)")
println("  Converged: $(converged)")
println("  Final μ: $(state_final.params.μ)")
println("  Final β: $(state_final.problem.β_heaviside)")

# Volume fraction achieved
s = (state_final.ctx.H * state_final.ctx.ϕ) ./ state_final.ctx.Hs
β = state_final.problem.β_heaviside
heaviside = state_final.problem.threshold_heaviside
denominator = tanh(β*heaviside) + tanh(β*(1 - heaviside))
ρ_phys = (@. (tanh(β*heaviside) + tanh(β*(s - heaviside))) / denominator)
active_volumes = state_final.ctx.volumes[ADMM.unmasked_elements(state_final.ctx.ϕ, state_final.problem.element_mask)]
total_volume = sum(active_volumes)
current_volume_frac = dot(ρ_phys[ADMM.unmasked_elements(state_final.ctx.ϕ, state_final.problem.element_mask)], active_volumes) / total_volume
println("\nConstraints:")
@printf("  Volume fraction: %.3f (target: %.3f)\n", actual_volume_frac, vol_frac)

# Stress statistics
max_stress = maximum(state_final.ctx.σ̃)
mean_stress = sum(state_final.ctx.σ̃) / length(state_final.ctx.σ̃)

@printf("  Max stress: %.4f (limit: %.4f)\n", max_stress, σ_lim)
@printf("  Mean stress: %.4f\n", mean_stress)

# Density statistics
ρ = state_final.ctx.ρ
n_solid = count(x -> x > 0.9, ρ)
n_void = count(x -> x < 0.1, ρ)
n_intermediate = length(ρ) - n_solid - n_void

println("\nDensity Distribution:")
println("  Solid (ρ>0.9): $(n_solid) / $(length(ρ)) ($(round(100*n_solid/length(ρ), digits=1))%)")
println("  Void (ρ<0.1): $(n_void) / $(length(ρ)) ($(round(100*n_void/length(ρ), digits=1))%)")
println("  Intermediate: $(n_intermediate) / $(length(ρ)) ($(round(100*n_intermediate/length(ρ), digits=1))%)")

# Compliance
compliance = dot(state_final.ctx.f, state_final.ctx.U)
@printf("\n  Compliance: %.6e\n", compliance)

println("\n" * "="^70)
println("DIAGNOSTIC CHECK")
println("="^70)
println("ctx.ϕ range: [$(minimum(state_final.ctx.ϕ)), $(maximum(state_final.ctx.ϕ))]")
println("ctx.ρ range: [$(minimum(state_final.ctx.ρ)), $(maximum(state_final.ctx.ρ))]")
println("ctx.σ̄ (von Mises) range: [$(minimum(state_final.ctx.σ̄)), $(maximum(state_final.ctx.σ̄))]")
println("ctx.σ̃ (relaxed) range: [$(minimum(state_final.ctx.σ̃)), $(maximum(state_final.ctx.σ̃))]")
println("ctx.α (auxiliary) range: [$(minimum(state_final.ctx.α)), $(maximum(state_final.ctx.α))]")
println("Unique values in ctx.σ̃: $(length(unique(state_final.ctx.σ̃)))")

println("\n" * "="^70)
println("Topology optimization complete!")
println("="^70)
println("\nDesign saved in: state_final.ctx.ρ")

# ============================================================================
# VISUALIZATION
# ============================================================================

println("\nGenerating visualization...")

# Reshape density to mesh grid
ρ_grid = reshape(state_final.ctx.ρ, nelx, nely)

# Create figure
fig = Figure(size=(1400, 600))

# Left plot: Optimized topology
ax1 = Axis(fig[1, 1],
           title="Optimized Topology (Density Distribution)",
           xlabel="x [m]",
           ylabel="y [m]",
           aspect=DataAspect())

hm = heatmap!(ax1, range(0, Lx, length=nelx+1), range(0, Ly, length=nely+1), ρ_grid,
              colormap=:grays,
              colorrange=(0, 1))

Colorbar(fig[1, 2], hm, label="Density ρ")

# Right plot: Binary design with stress coloring
ax2 = Axis(fig[1, 3],
           title="Binary Design with Stress (ρ > 0.5)",
           xlabel="x [m]",
           ylabel="y [m]",
           aspect=DataAspect())

# Create stress grid
σ̃_grid = reshape(state_final.ctx.σ̃, nelx, nely)

# Mask: show stress only for solid elements (ρ > 0.5), NaN for void
ρ_binary = ρ_grid .> problem.threshold_heaviside + 0.05
σ̃_binary = copy(σ̃_grid)
σ̃_binary[.!ρ_binary] .= NaN  # Set void elements to NaN

hm2 = heatmap!(ax2, range(0, Lx, length=nelx+1), range(0, Ly, length=nely+1), σ̃_binary,
         colormap=:turbo,
         colorrange=(0, σ_lim),
         nan_color=:black)

Colorbar(fig[1, 4], hm2, label="Stress σ̃ (solid elements)")

# Add boundary condition markers to both plots
for ax in [ax1, ax2]
    # Automatically detect and visualize clamped nodes
    clamped_dofs = Set(boundary_dofs)
    clamped_nodes = Set{Int}()
    
    for node_id in 1:length(mesh.nodes)
        node_dofs = FEM.get_node_dofs(mesh, node_id)
        if all(dof in clamped_dofs for dof in node_dofs)
            push!(clamped_nodes, node_id)
        end
    end
    
    # Show clamped nodes
    if !isempty(clamped_nodes)
        clamped_x = [mesh.nodes[node_id].coords[1] for node_id in clamped_nodes]
        clamped_y = [mesh.nodes[node_id].coords[2] for node_id in clamped_nodes]
        
        scatter!(ax, clamped_x, clamped_y, 
                 marker=:utriangle, markersize=6, color=:blue, 
                 strokewidth=1, strokecolor=:black, alpha=0.8)
    end
    
    # Show loaded nodes
    if !isempty(right_force_nodes)
        load_x = [mesh.nodes[node_id].coords[1] for node_id in right_force_nodes]
        load_y = [mesh.nodes[node_id].coords[2] for node_id in right_force_nodes]
        
        scatter!(ax, load_x, load_y, 
                 marker=:dtriangle, markersize=12, color=:red, 
                 strokewidth=2, strokecolor=:black, alpha=0.9)
    end
end

# Overall title with key metrics
Label(fig[0, :], 
      @sprintf("L shaped domain: Vol=%.1f%%, σmax=%.3f, Compliance=%.2e", 
               actual_volume_frac*100, max_stress, compliance),
      fontsize=20,
      font=:bold)

# Save figure
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)
output_file = joinpath(output_dir, "l_domain_result.png")
save(output_file, fig)

println("  Saved visualization to: $(output_file)")

# Display
# display(fig)

println("\nVisualization complete!")