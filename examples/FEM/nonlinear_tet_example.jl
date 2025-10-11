# Example: Nonlinear FEM analysis with tetrahedral elements and Neo-Hookean material
#
# This example demonstrates:
# 1. Creating a tetrahedral mesh
# 2. Using the Neo-Hookean hyperelastic material model
# 3. Setting up boundary conditions
# 4. Solving with Newton's method for large deformations

using AlgorithmEngineering.FEM
using LinearAlgebra
using SparseArrays

include(joinpath(@__DIR__, "visualization_utils.jl"))

# =============================================================================
# Problem setup
# =============================================================================

# Geometry
width = 1.0
height = 0.5
depth = 0.5

# Mesh density
nx, ny, nz = 4, 2, 2

# Material properties (Neo-Hookean)
E = 1e6      # Young's modulus (Pa)
ν = 0.45     # Poisson's ratio (nearly incompressible)

# =============================================================================
# Create mesh and material
# =============================================================================

# Generate tetrahedral mesh
mesh = generate_tetrahedral_mesh(width, height, depth, nx, ny, nz)
println("Generated mesh with $(length(mesh.nodes)) nodes and $(length(mesh.elements)) elements")

# Create Neo-Hookean material
material = NeoHookean(E, ν)
println("Material properties:")
props = get_material_properties(material)
println("  E = $(props.E), ν = $(props.ν)")
println("  μ = $(props.μ), λ = $(props.λ)")

# =============================================================================
# Boundary conditions and loads
# =============================================================================

# Fix left boundary (x = 0)
left_nodes = mesh.boundary_nodes["left"]
fixed_nodes = falses(length(mesh.nodes))
fixed_nodes[left_nodes] .= true

println("Fixed $(sum(fixed_nodes)) nodes on left boundary")

# Apply downward force on right boundary
right_nodes = mesh.boundary_nodes["right"]
num_dofs = length(mesh.nodes) * 3

f_ext = zeros(num_dofs)
force_magnitude = -1000.0  # Downward force in z-direction

for node_id in right_nodes
    dof_z = compute_global_dof(node_id, 3, mesh)
    f_ext[dof_z] = force_magnitude / length(right_nodes)
end

println("Applied total force: $(force_magnitude) N in z-direction on $(length(right_nodes)) nodes")

# =============================================================================
# Linear solution (for comparison)
# =============================================================================

println("\n" * "="^70)
println("Linear solution (small strain approximation)")
println("="^70)

# Create equivalent linear elastic material
linear_material = LinearElastic(E, ν)

# Assemble stiffness matrix
K = assemble_stiffness_matrix(mesh, linear_material, ThreeDimensional())
println("Assembled stiffness matrix: $(size(K))")

# Apply boundary conditions
active_dofs = .!repeat(fixed_nodes, inner=3)
active_indices = findall(active_dofs)

K_reduced = K[active_indices, active_indices]
f_reduced = f_ext[active_indices]

# Solve
u_linear_reduced = K_reduced \ f_reduced
u_linear = zeros(num_dofs)
u_linear[active_indices] = u_linear_reduced

# Compute maximum displacement
max_disp = maximum(abs.(u_linear))
println("Maximum displacement: $(max_disp) m")

# =============================================================================
# Nonlinear solution (Neo-Hookean with Newton's method)
# =============================================================================

println("\n" * "="^70)
println("Nonlinear solution (Neo-Hookean material, Newton's method)")
println("="^70)

# Note: This is a simplified demonstration
# For a full implementation, we would need to:
# 1. Compute tangent stiffness matrix at each Newton iteration
# 2. Update element stiffness based on current deformation
# 3. Use proper stress computation for large deformations

println("Note: Full nonlinear solver requires tangent stiffness computation")
println("      which depends on the current deformation gradient.")

# For demonstration, compute energy density and stress at different F values
println("\nStress-strain behavior:")
F_test = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]  # Identity (no deformation)
W0 = energy_density(material, F_test)
P0 = stress_tensor(material, F_test)
println("At F = I (no deformation):")
println("  Energy density W = $(W0)")
println("  Stress tensor P = $(P0)")

F_test2 = [1.1 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]  # 10% stretch in x
W1 = energy_density(material, F_test2)
P1 = stress_tensor(material, F_test2)
println("\nAt F = diag(1.1, 1.0, 1.0) (10% stretch in x):")
println("  Energy density W = $(W1)")
println("  Stress tensor P[1,1] = $(P1[1,1])")