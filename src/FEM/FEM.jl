# FEM.jl - Finite Element Method module
# this is a draft (FEM is a relatively standard implementation but also annoying to do, so
# the codebase is largely made using a combination of Codex and Claude for expediency)

"""
FEM

Finite Element Method module for topology optimization.
"""
module FEM

using LinearAlgebra, SparseArrays, StaticArrays

# Import submodules
include("core/mesh.jl")
include("core/materials.jl")
include("core/elements.jl")
include("core/assembly.jl")
include("core/boundary.jl")
include("core/solvers.jl")
include("core/analysis.jl")

# exports
# Element types (Quad4 for 2D, Hex8/Tet4 for 3D)
export Mesh, Node, Element, Quad4, Hex8, Tet4

# Mesh generation
export generate_rectangular_mesh, generate_hexahedral_mesh, generate_tetrahedral_mesh

# Mesh utilities
export get_nodes, get_elements, get_boundary_nodes, get_node_dofs, get_element_dofs
export get_dofs_per_node, compute_global_dof, get_node_local_dofs, get_element_coords

# Element computations
export compute_element_stiffness, shape_functions
export compute_deformation_gradient, compute_Dm_inv, compute_tet_volume

# Material models - Linear
export Material, LinearElastic, PlaneStress, PlaneStrain, ThreeDimensional
export get_constitutive_matrix, get_material_properties

# Material models - Nonlinear
export NonlinearMaterial, NeoHookean
export energy_density, stress_tensor, stress_differential

# Assembly
export assemble_stiffness_matrix, assemble_force_vector, apply_boundary_conditions

# Boundary conditions and loads
export BoundaryCondition, DirichletBC, NeumannBC, LoadCase
export apply_boundary_conditions!, apply_loads!, get_constrained_dofs
export create_fixed_support, create_roller_support, create_point_load
export collect_dirichlet_constraints, collect_neumann_loads

# Linear solvers
export solve_displacements

# Nonlinear solvers
export solve_newton, solve_nonlinear_displacements, compute_elastic_forces

# Stress and strain analysis
export compute_element_stresses, compute_nodal_stresses
export compute_strain_energy, compute_compliance, extract_displacements
export compute_von_mises_stress, compute_principal_stresses

end # module FEM
