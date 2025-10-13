"""
FEM MBB Beam 3D Analysis Example

This example demonstrates 3D finite element analysis of a Messerschmitt-Bölkow-Blohm
(MBB) beam using hexahedral elements.

Setup:
- 3D rectangular beam with 10:5:1 aspect ratio (length:height:thickness)
- Beam oriented with wide face on sides, narrow face on top
- Point load at the center of the top face (downward in z-direction)
- Left bottom edge: fixed support along entire edge (ux = uy = uz = 0)
- Right bottom edge: roller support along entire edge (uz = 0, ux and uy free)
- Uses Hex8 (8-node hexahedral) elements with 3D analysis
- Visualizes displacement field using GLMakie

The MBB beam is a classic benchmark problem in structural mechanics and topology
optimization. This is the 3D formulation.

Author: SATO.jl Contributors (adapted for ADMM.FEM)
"""

using AlgorithmEngineering.FEM
using LinearAlgebra
using SparseArrays
using GLMakie
using Printf

include(joinpath(@__DIR__, "visualization_utils.jl"))

"""
    visualize_mbb_fem_3d(mesh, displacements, load_node; scale=1.0, title="MBB Beam 3D FEM Analysis")

Create a visualization of 3D MBB beam deformation with load point marked.
"""
function visualize_mbb_fem_3d(mesh, displacements, load_node; scale=1.0, title="MBB Beam 3D FEM Analysis")
    data = compute_fem_visualization_data_3d(mesh, displacements; scale=scale)

    u_min = minimum(data.u_magnitude)
    u_max = maximum(data.u_magnitude)
    mean_u = sum(data.u_magnitude) / length(data.u_magnitude)

    println("  Displacement range: $(u_min * 1e6) to $(u_max * 1e6) μm")
    println("  Mean displacement: $(mean_u * 1e6) μm")

    fig = create_fem_deformation_figure_3d(data;
        title=title,
        colormap=:viridis,
        colorrange=(0.0, max(u_max, eps(Float64))),
        overlay_original=true,
        show_nodes=true,
        load_node=load_node)

    return fig
end

println("="^70)
println("FEM MBB Beam 3D Analysis")
println("="^70)

# Problem parameters - 3D beam dimensions
Lx = 6.0          # Beam length [m] (x-direction, horizontal)
Ly = 0.6          # Beam thickness [m] (y-direction, horizontal depth)
Lz = 3.0          # Beam height [m] (z-direction, vertical/upward)
nelx = 20         # Number of elements in x-direction
nely = 2          # Number of elements in y-direction (thin dimension)
nelz = 10         # Number of elements in z-direction (tall dimension)

# Material properties (steel)
E = 200e9         # Young's modulus [Pa]
ν = 0.3           # Poisson's ratio
ρ = 7850.0        # Density [kg/m³]

# Loading
load_magnitude = -10000.0  # 10 kN downward force [N]

println("\nProblem Setup:")
println("  Geometry: $(Lx) × $(Ly) × $(Lz) m (length × thickness × height)")
println("  Aspect ratio: $(Int(Lx/Ly)):1:$(Int(Lz/Ly)) (wide face on sides)")
println("  Mesh: $(nelx) × $(nely) × $(nelz) = $(nelx*nely*nelz) Hex8 elements")
println("  Material: E = $(E/1e9) GPa, ν = $(ν), ρ = $(ρ) kg/m³")
println("  Load: $(abs(load_magnitude)/1000) kN downward (z-direction) at top center")

# Generate 3D mesh
println("\nGenerating 3D hexahedral mesh...")
mesh = FEM.generate_hexahedral_mesh(Lx, Ly, Lz, nelx, nely, nelz)

nodes = FEM.get_nodes(mesh)
n_nodes = length(nodes)
n_dofs = 3 * n_nodes  # 3D DOFs

println("  Nodes: $(n_nodes)")
println("  DOFs: $(n_dofs)")
println("  Elements: $(length(mesh.elements))")

# Define material
material = FEM.LinearElastic(E, ν, ρ)
analysis_type = FEM.ThreeDimensional()

# Boundary conditions for 3D MBB beam
println("\nApplying boundary conditions...")

# Get boundary nodes
left_nodes = FEM.get_boundary_nodes(mesh, "left")
right_nodes = FEM.get_boundary_nodes(mesh, "right")
front_nodes = FEM.get_boundary_nodes(mesh, "front")  # Bottom face (z = 0)

# Find nodes along the left and right bottom edges
# Left bottom edge: all nodes at x=0, z=0 (varies in y)
left_bottom_edge = intersect(left_nodes, front_nodes)

# Right bottom edge: all nodes at x=Lx, z=0 (varies in y)
right_bottom_edge = intersect(right_nodes, front_nodes)

# Dirichlet boundary conditions
dirichlet_bcs = FEM.DirichletBC{Float64}[]

# Left bottom edge: Fixed support (ux = 0, uy = 0, uz = 0 for all nodes)
# This prevents rigid body motion in all directions
append!(dirichlet_bcs, FEM.create_fixed_support(mesh, left_bottom_edge))

# Right bottom edge: Roller support (uz = 0, ux and uy free)
# This prevents vertical motion while allowing horizontal expansion in x and y
for node_id in right_bottom_edge
    append!(dirichlet_bcs, FEM.create_roller_support(mesh, [node_id], 3))  # uz = 0 (vertical constraint)
end

fixed_dofs, _ = FEM.collect_dirichlet_constraints(mesh, dirichlet_bcs)

println("  Left bottom edge: $(length(left_bottom_edge)) nodes fixed (ux = uy = uz = 0)")
println("    Edge from $(nodes[left_bottom_edge[1]].coords) to $(nodes[left_bottom_edge[end]].coords)")
println("  Right bottom edge: $(length(right_bottom_edge)) nodes as rollers (uz = 0, ux and uy free)")
println("    Edge from $(nodes[right_bottom_edge[1]].coords) to $(nodes[right_bottom_edge[end]].coords)")
println("  Total fixed DOFs: $(length(fixed_dofs))")

# Apply load at center of top face (z = Lz, "back" boundary)
# Find the node closest to the geometric center (x=Lx/2, y=Ly/2, z=Lz)
top_nodes = FEM.get_boundary_nodes(mesh, "back")  # Top face is at z = Lz
center_x = Lx / 2
center_y = Ly / 2
center_z = Lz

# Find node closest to center of top face
min_dist = Inf
load_node = top_nodes[1]
for nid in top_nodes
    node = nodes[nid]
    dx = node.coords[1] - center_x
    dy = node.coords[2] - center_y
    dist = sqrt(dx^2 + dy^2)
    if dist < min_dist
        min_dist = dist
        load_node = nid
    end
end

load_coords = nodes[load_node].coords
load_dof = FEM.get_node_dofs(mesh, load_node)[3]  # z-direction (downward)

load_case = FEM.LoadCase{Float64}()
append!(load_case.dirichlet_bcs, dirichlet_bcs)
merge!(load_case.point_loads, FEM.create_point_load(mesh, load_node, 3, load_magnitude))  # direction 3 = z

println("  Load: node $(load_node) at $(load_coords), DOF $(load_dof) = $(load_magnitude) N (downward in z)")
println("    (Center of top face: x=$(center_x), y=$(center_y), z=$(center_z))")

# Assemble global stiffness matrix
println("\nAssembling global stiffness matrix...")
K = FEM.assemble_stiffness_matrix(mesh, material, analysis_type)

println("  Matrix size: $(size(K, 1)) × $(size(K, 2))")
println("  Non-zero entries: $(nnz(K))")
println("  Sparsity: $(round(100 * (1 - nnz(K) / prod(size(K))), digits=2))%")
println("  Memory: $(round(nnz(K) * 8 / 1024^2, digits=2)) MB")

# Assemble force vector
println("\nAssembling force vector...")
F = zeros(Float64, n_dofs)
FEM.apply_loads!(F, mesh, load_case)
F_original = copy(F)

# Apply boundary conditions
println("\nApplying boundary conditions to system...")
K_bc, F_bc = FEM.apply_boundary_conditions!(K, F, mesh, load_case.dirichlet_bcs)

# Check condition number (optional diagnostic)
if n_dofs < 10000  # Only for smaller systems
    κ = cond(Matrix(K_bc))
    println("  Condition number: $(round(κ, sigdigits=3))")
end

# Solve for displacements
println("\nSolving linear system K*u = F...")
u = FEM.solve_displacements(K_bc, F_bc)

# Compute solution statistics
max_displacement = maximum(abs.(u))
u_x = [u[nodes[i].dofs[1]] for i in 1:n_nodes]
u_y = [u[nodes[i].dofs[2]] for i in 1:n_nodes]
u_z = [u[nodes[i].dofs[3]] for i in 1:n_nodes]
max_u_x = maximum(abs.(u_x))
max_u_y = maximum(abs.(u_y))
max_u_z = maximum(abs.(u_z))

println("\nSolution Statistics:")
@printf("  Maximum displacement: %.4f mm\n", 1000*max_displacement)
@printf("  Maximum x-displacement: %.4f mm\n", 1000*max_u_x)
@printf("  Maximum y-displacement: %.4f mm\n", 1000*max_u_y)
@printf("  Maximum z-displacement: %.4f mm\n", 1000*max_u_z)

# Center deflection
center_deflection = abs(u[load_dof])
@printf("  Center deflection: %.4f mm\n", 1000*center_deflection)

# Compute compliance and strain energy
compliance = FEM.compute_compliance(F_original, u)
strain_energy = FEM.compute_strain_energy(K, u)

@printf("  Compliance: %.6e J\n", compliance)
@printf("  Strain energy: %.6e J\n", strain_energy)

# Reaction forces at supports
R = K * u - F_original

# Sum reactions along the left and right bottom edges
R_left_y = sum(R[FEM.get_node_dofs(mesh, nid)[2]] for nid in left_bottom_edge)
R_right_y = sum(R[FEM.get_node_dofs(mesh, nid)[2]] for nid in right_bottom_edge)

# Total vertical reaction (should equal applied load)
R_total_y = R_left_y + R_right_y

println("\nReaction Forces:")
@printf("  Left edge (y-direction): %.2f N distributed over %d nodes\n", 
        R_left_y, length(left_bottom_edge))
@printf("  Right edge (y-direction): %.2f N distributed over %d nodes\n", 
        R_right_y, length(right_bottom_edge))
@printf("  Sum of vertical reactions: %.2f N (applied load: %.2f N)\n", 
        R_total_y, load_magnitude)

# Check equilibrium (reactions should balance applied load)
vertical_error = abs(R_total_y + load_magnitude)

println("\nEquilibrium Check:")
@printf("  Vertical force error: %.2e N\n", vertical_error)
@printf("  Equilibrium satisfied: %s\n", vertical_error < abs(load_magnitude) * 1e-6 ? "Yes" : "No")

# 3D Analysis Notes
println("\n3D Analysis Notes:")
println("  • Using 3D Hex8 elements")
println("  • Full 3D stress-strain relationships")
println("  • 3 DOFs per node (x, y, z displacements)")
println("  • 3D boundary conditions and loading")

# Visualization
println("\nGenerating 3D visualization...")

# Automatic scaling: scale displacements so max is ~8% of beam length
target_display_deflection = 0.08 * Lx
displacement_scale = target_display_deflection / max_displacement

@printf("  Displacement scale factor: %.1f\n", displacement_scale)

# Create visualization
fig = visualize_mbb_fem_3d(mesh, u, load_node;
                                scale=displacement_scale,
                                title="MBB Beam 3D - Finite Element Analysis")

# Save figure
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)
output_file = joinpath(output_dir, "fem_mbb_beam_3d.png")
save(output_file, fig)
println("  Saved visualization to: $(output_file)")

# Display the figure
display(fig)

println("\n" * "="^70)
println("3D Analysis complete!")
println("="^70)
println("\nKey Results Summary:")
@printf("  • Center deflection: %.4f mm\n", 1000*center_deflection)
@printf("  • Maximum displacement: %.4f mm\n", 1000*max_displacement)
@printf("  • Compliance: %.4e J\n", compliance)
println("  • Equilibrium satisfied: $(vertical_error < abs(load_magnitude) * 1e-6)")
println("  • Analysis method: 3D with Hex8 elements")
println("\nVisualization saved to: $(output_file)")
