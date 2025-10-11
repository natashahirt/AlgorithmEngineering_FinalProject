# FEM.jl - Finite Element Method module
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
include("core/analysis.jl")

# exports
export Mesh, Node, Element, Quad4, Hex8
export generate_rectangular_mesh, generate_hexahedral_mesh
export get_nodes, get_elements, get_boundary_nodes, get_node_dofs, get_element_dofs
export get_dofs_per_node, compute_global_dof, get_node_local_dofs
export compute_element_stiffness, shape_functions
export Material, LinearElastic, PlaneStress, PlaneStrain, ThreeDimensional
export get_constitutive_matrix, get_material_properties
export assemble_stiffness_matrix, assemble_force_vector, apply_boundary_conditions
export BoundaryCondition, DirichletBC, NeumannBC, LoadCase
export apply_boundary_conditions!, apply_loads!, get_constrained_dofs
export create_fixed_support, create_roller_support, create_point_load
export collect_dirichlet_constraints, collect_neumann_loads
export solve_displacements, compute_element_stresses, compute_nodal_stresses
export compute_strain_energy, compute_compliance, extract_displacements
export compute_von_mises_stress, compute_principal_stresses

end # module FEM
