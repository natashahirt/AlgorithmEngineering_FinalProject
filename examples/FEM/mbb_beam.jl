"""
FEM MBB Beam Visualization Example

This example demonstrates finite element analysis of a Messerschmitt-Bölkow-Blohm
(MBB) beam, showing both the initial (undeformed) state and the deformed shape
under load.

Setup:
- 2D rectangular beam with 2:1 aspect ratio
- Point load at the center of the top edge
- Left corner: pin support (constrained in x and y)
- Right corner: roller support (constrained in y-direction only)
- Uses Quad4 elements with plane stress assumption
- Visualizes displacement field using GLMakie

The MBB beam is a classic benchmark problem in structural mechanics and topology
optimization. This is the full-domain formulation.

Author: SATO.jl Contributors
"""

using AlgorithmEngineering.FEM
using LinearAlgebra
using SparseArrays
using GLMakie
using Printf

include(joinpath(@__DIR__, "visualization_utils.jl"))

"""
    visualize_mbb_fem(mesh, displacements; scale=1.0, title="MBB Beam FEM Analysis")

Create a side-by-side visualization of undeformed and deformed MBB beam.

# Arguments
- `mesh`: FEM mesh structure
- `displacements`: Displacement vector from FE analysis
- `scale`: Scaling factor for displacement visualization
- `title`: Overall plot title
"""
function visualize_mbb_fem(mesh, displacements; scale=1.0, title="MBB Beam FEM Analysis")
    data = compute_fem_visualization_data(mesh, displacements; scale=scale)

    u_min = minimum(data.u_magnitude)
    u_max = maximum(data.u_magnitude)
    mean_u = sum(data.u_magnitude) / length(data.u_magnitude)

    println("  Displacement range: $(u_min*1e6) to $(u_max*1e6) μm")
    println("  Mean displacement: $(mean_u*1e6) μm")

    left_nodes = FEM.get_boundary_nodes(mesh, "left")
    right_nodes = FEM.get_boundary_nodes(mesh, "right")

    left_bottom_node = left_nodes[1]
    right_bottom_node = right_nodes[1]

    annotate_bc = (ax, data) -> begin
        scatter!(ax,
                 [data.x_orig[left_bottom_node]],
                 [data.y_orig[left_bottom_node]],
                 marker=:rect,
                 markersize=12,
                 color=:red)

        scatter!(ax,
                 [data.x_orig[right_bottom_node]],
                 [data.y_orig[right_bottom_node]],
                 marker=:utriangle,
                 markersize=15,
                 color=:red)
    end

    fig = create_fem_deformation_figure(data;
                                        figure_size=(1600, 700),
                                        title=title,
                                        colormap=:viridis,
                                        colorrange=(0.0, u_max),
                                        overlay_original=true,
                                        annotate_undeformed=annotate_bc,
                                        colorbar_label="Displacement Magnitude [m]")

    return fig
end

println("="^70)
println("FEM MBB Beam Analysis and Visualization")
println("="^70)

# Problem parameters
Lx = 6.0          # Beam length [m]
Ly = 3.0          # Beam height [m] (2:1 aspect ratio)
nelx = 60         # Number of elements in x-direction
nely = 30         # Number of elements in y-direction
thickness = 0.1   # Beam thickness [m]

# Material properties (steel)
E = 200e9         # Young's modulus [Pa]
ν = 0.3           # Poisson's ratio
ρ = 7850.0        # Density [kg/m³]

# Loading
load_magnitude = -10000.0  # 10 kN downward force [N]

println("\nProblem Setup:")
println("  Geometry: $(Lx) × $(Ly) m (aspect ratio 2:1)")
println("  Thickness: $(thickness) m")
println("  Mesh: $(nelx) × $(nely) = $(nelx*nely) elements")
println("  Material: E = $(E/1e9) GPa, ν = $(ν), ρ = $(ρ) kg/m³")
println("  Load: $(abs(load_magnitude)/1000) kN downward at top center")

# Generate mesh
println("\nGenerating mesh...")
mesh = FEM.generate_rectangular_mesh(Lx, Ly, nelx, nely)

nodes = FEM.get_nodes(mesh)
n_nodes = length(nodes)
n_dofs = 2 * n_nodes

println("  Nodes: $(n_nodes)")
println("  DOFs: $(n_dofs)")

# Define material
material = FEM.LinearElastic(E, ν, ρ)
analysis_type = FEM.PlaneStress()

# Boundary conditions for MBB beam (full domain)
# Two roller supports at bottom corners
println("\nApplying boundary conditions...")

# Get boundary nodes
left_nodes = FEM.get_boundary_nodes(mesh, "left")
right_nodes = FEM.get_boundary_nodes(mesh, "right")
bottom_nodes = FEM.get_boundary_nodes(mesh, "bottom")

fixed_dofs = Int[]

# Left bottom corner: roller support (uy = 0, ux = 0 to prevent rigid body motion)
left_bottom_node = left_nodes[1]
left_dofs = FEM.get_node_dofs(mesh, left_bottom_node)
push!(fixed_dofs, left_dofs[1])  # ux = 0 (prevent horizontal rigid body motion)
push!(fixed_dofs, left_dofs[2])  # uy = 0 (vertical support)

# Right bottom corner: roller support (uy = 0, ux free)
right_bottom_node = right_nodes[1]
right_dof_y = FEM.get_node_dofs(mesh, right_bottom_node)[2]
push!(fixed_dofs, right_dof_y)

println("  Left corner (pin): node $(left_bottom_node), ux = 0, uy = 0")
println("  Right corner (roller): node $(right_bottom_node), uy = 0")
println("  Total fixed DOFs: $(length(fixed_dofs))")

# Apply load at center of top edge
top_nodes = FEM.get_boundary_nodes(mesh, "top")
center_idx = div(length(top_nodes) + 1, 2)
load_node = top_nodes[center_idx]
load_dof = FEM.get_node_dofs(mesh, load_node)[2]  # y-direction

loads = Dict(load_dof => load_magnitude)

println("  Load: node $(load_node), DOF $(load_dof) = $(load_magnitude) N")

# Assemble global stiffness matrix
println("\nAssembling global stiffness matrix...")
K = FEM.assemble_stiffness_matrix(mesh, material, analysis_type; thickness=thickness)

println("  Matrix size: $(size(K, 1)) × $(size(K, 2))")
println("  Non-zero entries: $(nnz(K))")
println("  Sparsity: $(round(100 * (1 - nnz(K) / prod(size(K))), digits=2))%")
println("  Memory: $(round(nnz(K) * 8 / 1024^2, digits=2)) MB")

# Assemble force vector
println("\nAssembling force vector...")
F = FEM.assemble_force_vector(loads, n_dofs)

# Apply boundary conditions
println("\nApplying boundary conditions to system...")
K_bc, F_bc = FEM.apply_boundary_conditions(K, F, fixed_dofs)

# Check condition number (optional diagnostic)
if n_dofs < 10000  # Only for smaller systems
    κ = cond(Matrix(K_bc))
    println("  Condition number: $(round(κ, sigdigits=3))")
end

# Solve for displacements
println("\nSolving linear system K*u = F...")
u = K_bc \ F_bc

# Compute solution statistics
max_displacement = maximum(abs.(u))
u_x = [u[nodes[i].dofs[1]] for i in 1:n_nodes]
u_y = [u[nodes[i].dofs[2]] for i in 1:n_nodes]
max_u_x = maximum(abs.(u_x))
max_u_y = maximum(abs.(u_y))

println("\nSolution Statistics:")
@printf("  Maximum displacement: %.4f mm\n", 1000*max_displacement)
@printf("  Maximum x-displacement: %.4f mm\n", 1000*max_u_x)
@printf("  Maximum y-displacement: %.4f mm\n", 1000*max_u_y)

# Center deflection
center_deflection = abs(u[load_dof])
@printf("  Center deflection: %.4f mm\n", 1000*center_deflection)

# Compute compliance and strain energy
compliance = dot(F, u)
strain_energy = 0.5 * dot(u, K * u)

@printf("  Compliance: %.6e J\n", compliance)
@printf("  Strain energy: %.6e J\n", strain_energy)

# Reaction forces at supports
R = K * u - F

# Reactions at left and right roller supports
left_nodes = FEM.get_boundary_nodes(mesh, "left")
right_nodes = FEM.get_boundary_nodes(mesh, "right")

left_bottom_node = left_nodes[1]
right_bottom_node = right_nodes[1]

R_left_y = R[FEM.get_node_dofs(mesh, left_bottom_node)[2]]
R_right_y = R[FEM.get_node_dofs(mesh, right_bottom_node)[2]]

# Total vertical reaction (should equal applied load)
R_total_y = R_left_y + R_right_y

println("\nReaction Forces:")
@printf("  Left support (y-direction): %.2f N\n", R_left_y)
@printf("  Right support (y-direction): %.2f N\n", R_right_y)
@printf("  Sum of vertical reactions: %.2f N (applied load: %.2f N)\n", 
        R_total_y, load_magnitude)

# Check equilibrium (reactions should balance applied load)
# Note: load_magnitude is negative (downward), reactions are positive (upward)
vertical_error = abs(R_total_y + load_magnitude)  # Should sum to zero

println("\nEquilibrium Check:")
@printf("  Vertical force error: %.2e N\n", vertical_error)
@printf("  Equilibrium satisfied: %s\n", vertical_error < abs(load_magnitude) * 1e-6 ? "Yes" : "No")

# Visualization
println("\nGenerating visualization...")

# Automatic scaling: scale displacements so max is ~8% of beam length
target_display_deflection = 0.08 * Lx
displacement_scale = target_display_deflection / max_displacement

@printf("  Displacement scale factor: %.1f\n", displacement_scale)

# Create visualization
fig = visualize_mbb_fem(mesh, u;
                        scale=displacement_scale,
                        title="MBB Beam - Finite Element Analysis")

# Save figure
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)
output_file = joinpath(output_dir, "fem_mbb_beam_deformation.png")
save(output_file, fig)
println("  Saved visualization to: $(output_file)")

# Display the figure
display(fig)

println("\n" * "="^70)
println("Analysis complete!")
println("="^70)
println("\nKey Results Summary:")
@printf("  • Center deflection: %.4f mm\n", 1000*center_deflection)
@printf("  • Maximum displacement: %.4f mm\n", 1000*max_displacement)
@printf("  • Compliance: %.4e J\n", compliance)
println("  • Equilibrium satisfied: $(vertical_error < abs(load_magnitude) * 1e-6)")
println("\nVisualization saved to: $(output_file)")

return mesh, u, fig
