# boundary.jl - boundary conditions
"""
Boundary conditions and load application.
"""

"""
BoundaryCondition

Abstract type for boundary conditions.
"""
abstract type BoundaryCondition end

"""
DirichletBC

Dirichlet boundary condition specifying displacements.
"""
struct DirichletBC{T} <: BoundaryCondition
    node_ids::Vector{Int}
    direction::Int
    value::T
end

"""
NeumannBC

Neumann boundary condition specifying forces/tractions.
"""
struct NeumannBC{T} <: BoundaryCondition
    node_ids::Vector{Int}
    direction::Int
    value::T
end

"""
LoadCase

Container for all boundary conditions and loads.
"""
Base.@kwdef mutable struct LoadCase{T}
    dirichlet_bcs::Vector{DirichletBC{T}} = DirichletBC{T}[]
    neumann_bcs::Vector{NeumannBC{T}} = NeumannBC{T}[]
    point_loads::Dict{Int,T} = Dict{Int,T}()
    distributed_loads::Vector{Any} = Any[]  # Placeholder for distributed loads
end

"""
apply_boundary_conditions!(K, f, mesh, bcs)

Apply Dirichlet boundary conditions defined on the mesh to the system matrices.
This is a convenience wrapper around `assemble.apply_boundary_conditions`.
"""
function apply_boundary_conditions!(K, f, mesh::Mesh, bcs::Vector{DirichletBC{T}}) where {T}
    dofs, values = collect_dirichlet_constraints(mesh, bcs)
    values_converted = convert(Vector{eltype(f)}, values)
    return apply_boundary_conditions(K, f, dofs, values_converted)
end

"""
apply_loads!(f, load_case::LoadCase)

Apply all loads to force vector.
"""
function apply_loads!(f, mesh::Mesh, load_case::LoadCase{T}) where {T}
    loads = collect_neumann_loads(mesh, load_case.neumann_bcs)
    merge!(loads, load_case.point_loads) do _, existing, new
        return existing + new
    end

    for (dof, magnitude) in loads
        1 <= dof <= length(f) || continue
        f[dof] += convert(eltype(f), magnitude)
    end

    return f
end

"""
get_constrained_dofs(mesh, bcs)

Return the sorted list of constrained global DOF indices for the mesh.
"""
function get_constrained_dofs(mesh::Mesh, bcs::Vector{DirichletBC{T}}) where {T}
    dofs, _ = collect_dirichlet_constraints(mesh, bcs)
    return sort(unique(dofs))
end

# Utility functions for creating common boundary conditions

"""
create_fixed_support(mesh, node_ids)

Create Dirichlet boundary conditions that fix all translational DOFs for the
specified nodes.
"""
function create_fixed_support(mesh::Mesh, node_ids::Vector{Int})
    dofs_per_node = get_dofs_per_node(mesh)
    bcs = DirichletBC{Float64}[]

    for node_id in node_ids
        for direction in 1:dofs_per_node
            push!(bcs, DirichletBC{Float64}([node_id], direction, 0.0))
        end
    end

    return bcs
end

"""
create_roller_support(mesh, node_ids, direction)

Constrain a single direction for the supplied nodes.
"""
function create_roller_support(mesh::Mesh, node_ids::Vector{Int}, direction::Int)
    dofs_per_node = get_dofs_per_node(mesh)
    1 <= direction <= dofs_per_node ||
        error("Direction $direction not valid for $(dofs_per_node)D mesh")

    return [DirichletBC{Float64}(node_ids, direction, 0.0)]
end

"""
create_point_load(mesh, node_id, direction, force)

Return a dictionary entry mapping the appropriate global DOF to the provided
force value.
"""
function create_point_load(mesh::Mesh, node_id::Int, direction::Int, force::Real)
    dofs_per_node = get_dofs_per_node(mesh)
    1 <= direction <= dofs_per_node ||
        error("Direction $direction not valid for $(dofs_per_node)D mesh")

    global_dof = compute_global_dof(node_id, direction, mesh)
    return Dict{Int,Float64}(global_dof => Float64(force))
end

function create_point_load(::Type{T}, mesh::Mesh, node_id::Int, direction::Int, force) where {T<:Real}
    dofs_per_node = get_dofs_per_node(mesh)
    1 <= direction <= dofs_per_node ||
        error("Direction $direction not valid for $(dofs_per_node)D mesh")

    global_dof = compute_global_dof(node_id, direction, mesh)
    return Dict{Int,T}(global_dof => convert(T, force))
end

"""
collect_dirichlet_constraints(mesh, bcs)

Expand Dirichlet boundary conditions into global DOF indices and values.
"""
function collect_dirichlet_constraints(mesh::Mesh, bcs::Vector{DirichletBC{T}}) where {T}
    dofs = Int[]
    values = T[]

    for bc in bcs
        for node_id in bc.node_ids
            global_dof = compute_global_dof(node_id, bc.direction, mesh)
            push!(dofs, global_dof)
            push!(values, bc.value)
        end
    end

    return dofs, values
end

"""
collect_neumann_loads(mesh, bcs)

Convert Neumann boundary conditions into a dictionary that maps global DOFs to
load magnitudes.
"""
function collect_neumann_loads(mesh::Mesh, bcs::Vector{NeumannBC{T}}) where {T}
    loads = Dict{Int,T}()
    for bc in bcs
        for node_id in bc.node_ids
            global_dof = compute_global_dof(node_id, bc.direction, mesh)
            loads[global_dof] = get(loads, global_dof, zero(T)) + bc.value
        end
    end
    return loads
end
