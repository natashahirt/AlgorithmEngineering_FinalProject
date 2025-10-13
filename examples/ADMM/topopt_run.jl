"""
Minimal Topology Optimization Example - MBB Beam

2D beam with stress-constrained compliance minimization using ADMM.
Based on the classic Messerschmitt-Bölkow-Blohm (MBB) benchmark.

Setup:
- Small 10×5 mesh for speed
- Volume fraction: 50%
- Stress limit: enforced via ADMM
"""

using Pkg; Pkg.activate(dirname(dirname(@__DIR__)))

using AlgorithmEngineering.ADMM
using AlgorithmEngineering.FEM
using StaticArrays
using LinearAlgebra
using Printf
using GLMakie

println("="^70)
println("Topology Optimization: MBB Beam (ADMM)")
println("="^70)

# ============================================================================
# PROBLEM SETUP
# ============================================================================

# Geometry (aspect ratio 2:1)
Lx = 6.0          # Length [m]
Ly = 3.0          # Height [m]
nelx = 10         # Elements in x (keep small for speed)
nely = 5          # Elements in y

# Material
E = 1.0           # Young's modulus (normalized)
ν = 0.3           # Poisson's ratio

# Optimization parameters
vol_frac = 0.5    # Target volume fraction (50% material)
σ_lim = 0.1       # Stress limit

println("\nProblem Setup:")
println("  Mesh: $(nelx)×$(nely) = $(nelx*nely) elements")
println("  Volume fraction: $(vol_frac*100)%")
println("  Stress limit: $(σ_lim)")

# ============================================================================
# MESH & BOUNDARY CONDITIONS
# ============================================================================

println("\nGenerating mesh...")
mesh = FEM.generate_rectangular_mesh(Lx, Ly, nelx, nely)

# Get boundary nodes
left_nodes = FEM.get_boundary_nodes(mesh, "left")
right_nodes = FEM.get_boundary_nodes(mesh, "right")
top_nodes = FEM.get_boundary_nodes(mesh, "top")

# Boundary conditions: MBB beam (symmetric half)
# Left bottom corner: pin (ux=0, uy=0)
left_bottom_node = left_nodes[1]
left_dofs = FEM.get_node_dofs(mesh, left_bottom_node)

# Right bottom corner: roller (uy=0)
right_bottom_node = right_nodes[1]
right_dof_y = FEM.get_node_dofs(mesh, right_bottom_node)[2]

boundary_dofs = [left_dofs[1], left_dofs[2], right_dof_y]

println("  Boundary: pin at left bottom, roller at right bottom")

# Loading: downward force at center of top edge
center_idx = div(length(top_nodes) + 1, 2)
load_node = top_nodes[center_idx]

forces = Dict(load_node => SVector(0.0, -1.0))  # 1N downward

println("  Load: 1N downward at top center (node $(load_node))")

# ============================================================================
# TOPOLOGY OPTIMIZATION PROBLEM
# ============================================================================

println("\nCreating topology optimization problem...")

material = FEM.LinearElastic(E, ν, 1.0)  # E, ν, density

problem = ADMM.TopOptProblem(
    mesh = mesh,
    material = material,
    analysis_type = FEM.PlaneStress(),
    forces = forces,
    boundary_dofs = boundary_dofs,
    vol_frac = vol_frac,
    σ_lim = σ_lim,
    r_filter = 1.5,           # Filter radius (1.5 × element size)
    β_heaviside = 1.0,        # Heaviside sharpness (starts at 1.0)
    η_heaviside = 0.5,        # Heaviside threshold
    max_iter_mma = 50,        # MMA inner iterations
    mma_tol = 1e-3            # MMA convergence tolerance
)

# ============================================================================
# ADMM SOLVER
# ============================================================================

println("\nInitializing ADMM solver...")

admm_params = ADMM.ADMMParams(
    μ = 1.0,                  # Penalty parameter
    reltol = 1e-2,            # Relative tolerance
    abstol = 1e-3,            # Absolute tolerance
    adaptive_μ = true         # Enable adaptive penalty
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

state_final, iters, converged = ADMM.run_admm!(state, max_iter=100, verbose=true)

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
volumes = state_final.ctx.volumes
total_volume = sum(volumes)
actual_volume_frac = dot(state_final.ctx.ρ, volumes) / total_volume

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
fig = Figure(size=(1200, 600))

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

# Right plot: Binary design (thresholded)
ax2 = Axis(fig[1, 3],
           title="Binary Design (ρ > 0.5)",
           xlabel="x [m]",
           ylabel="y [m]",
           aspect=DataAspect())

ρ_binary = ρ_grid .> 0.5

heatmap!(ax2, range(0, Lx, length=nelx+1), range(0, Ly, length=nely+1), ρ_binary,
         colormap=:grays,
         colorrange=(0, 1))

# Add boundary condition markers to both plots
for ax in [ax1, ax2]
    # Left bottom: Pin support (fixed in x and y) - shown as triangle
    scatter!(ax, [0.0], [0.0], marker=:utriangle, markersize=20, color=:blue, strokewidth=2, strokecolor=:black)
    # Right bottom: Roller support (fixed in y only) - shown as circle
    scatter!(ax, [Lx], [0.0], marker=:circle, markersize=15, color=:blue, strokewidth=2, strokecolor=:black)
    # Load point at top center - shown as downward arrow
    scatter!(ax, [Lx/2], [Ly], marker=:dtriangle, markersize=20, color=:red, strokewidth=2, strokecolor=:black)
end

# Overall title with key metrics
Label(fig[0, :], 
      @sprintf("MBB Beam: Vol=%.1f%%, σmax=%.3f, Compliance=%.2e", 
               actual_volume_frac*100, max_stress, compliance),
      fontsize=20,
      font=:bold)

# Save figure
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)
output_file = joinpath(output_dir, "topopt_mbb_result.png")
save(output_file, fig)

println("  Saved visualization to: $(output_file)")

# Display
display(fig)

println("\nVisualization complete!")
