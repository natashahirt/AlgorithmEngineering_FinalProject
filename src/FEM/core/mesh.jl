# mesh.jl - mesh data structures
"""
Mesh data structures and generation utilities.
"""

"""
Node{dim,T}

Node in finite element mesh.
"""
struct Node{dim,T}
    id::Int
    coords::SVector{dim,T}
    dofs::Vector{Int}
end

"""
Element{ElementType,T}

Abstract type for finite elements.
"""
abstract type Element{ElementType,T} end

"""
Quad4{T}

4-node quadrilateral element for 2D analysis.
"""
struct Quad4{T} <: Element{:Quad4,T}
    id::Int
    nodes::SVector{4,Int}
    material_id::Int
end

"""
Hex8{T}

8-node hexahedral element for 3D structured mesh analysis.
"""
struct Hex8{T} <: Element{:Hex8,T}
    id::Int
    nodes::SVector{8,Int}
    material_id::Int
end

"""
Tet4{T}

4-node tetrahedral element for 3D finite element analysis.
Preferred for nonlinear analysis and complex geometries.
"""
struct Tet4{T} <: Element{:Tet4,T}
    id::Int
    nodes::SVector{4,Int}
    material_id::Int
end

"""
Mesh{dim,T,ElementType}

Finite element mesh with nodes and elements.
"""
struct Mesh{dim,T,ElementType}
    nodes::Vector{Node{dim,T}}
    elements::Vector{Element{ElementType,T}}
    boundary_nodes::Dict{String,Vector{Int}}
end

"""
generate_rectangular_mesh(width, height, nx, ny; element_type=:Quad4)

Generate structured rectangular mesh.
"""
function generate_rectangular_mesh(width, height, nx, ny; element_type::Symbol = :Quad4)
    element_type == :Quad4 || error("Only Quad4 elements supported")
    nx > 0 && ny > 0 || error("nx and ny must be positive")

    hx = width / nx
    hy = height / ny

    nnx = nx + 1
    nny = ny + 1
    total_nodes = nnx * nny

    nodes = Vector{Node{2,Float64}}(undef, total_nodes)
    left = Int[]
    right = Int[]
    bottom = Int[]
    top = Int[]

    for j in 0:ny
        y = j * hy
        for i in 0:nx
            x = i * hx
            nid = j * nnx + i + 1
            coords = SVector{2,Float64}(x, y)
            dofs = [2 * (nid - 1) + 1, 2 * (nid - 1) + 2]
            nodes[nid] = Node{2,Float64}(nid, coords, dofs)

            if i == 0
                push!(left, nid)
            elseif i == nx
                push!(right, nid)
            end

            if j == 0
                push!(bottom, nid)
            elseif j == ny
                push!(top, nid)
            end
        end
    end

    num_elements = nx * ny
    elements = Vector{Element{:Quad4,Float64}}(undef, num_elements)

    eid = 1
    for j in 1:ny
        row_offset = (j - 1) * nnx
        next_row_offset = j * nnx
        for i in 1:nx
            n1 = row_offset + i
            n2 = row_offset + i + 1
            n3 = next_row_offset + i + 1
            n4 = next_row_offset + i
            elements[eid] = Quad4{Float64}(eid, SVector{4,Int}(n1, n2, n3, n4), 1)
            eid += 1
        end
    end

    boundary_nodes = Dict(
        "left" => left,
        "right" => right,
        "bottom" => bottom,
        "top" => top,
    )

    return Mesh{2,Float64,:Quad4}(nodes, elements, boundary_nodes)
end

"""
get_nodes(mesh::Mesh)

Get all nodes from mesh.
"""
get_nodes(mesh::Mesh) = mesh.nodes

"""
get_elements(mesh::Mesh)

Get all elements from mesh.
"""
get_elements(mesh::Mesh) = mesh.elements

"""
get_boundary_nodes(mesh::Mesh, boundary_name::String)

Get nodes on named boundary.
"""
function get_boundary_nodes(mesh::Mesh, boundary_name::String)
    return get(mesh.boundary_nodes, boundary_name, Int[])
end

"""
get_node_dofs(mesh::Mesh, node_id::Int)

Get global DOF indices for node.
"""
function get_node_dofs(mesh::Mesh, node_id::Int)
    return mesh.nodes[node_id].dofs
end

get_node_dofs(node::Node) = node.dofs

"""
get_element_dofs(mesh::Mesh, element)

Get global DOF indices for element.
"""
function get_element_dofs(mesh::Mesh, element)
    dofs = Int[]
    for nid in element.nodes
        append!(dofs, mesh.nodes[nid].dofs)
    end
    return dofs
end

# =============================================================================
# 3D Mesh Generation
# =============================================================================

"""
generate_hexahedral_mesh(length, width, height, nx, ny, nz)

Generate a structured 3D hexahedral mesh.

# Arguments
- `length::Real`: Length in x-direction
- `width::Real`: Width in y-direction  
- `height::Real`: Height in z-direction
- `nx::Int`: Number of elements in x-direction
- `ny::Int`: Number of elements in y-direction
- `nz::Int`: Number of elements in z-direction

# Returns
- `Mesh`: Generated 3D finite element mesh
"""
function generate_hexahedral_mesh(length, width, height, nx, ny, nz)
    nx > 0 && ny > 0 && nz > 0 || error("nx, ny, nz must be positive")

    hx = length / nx
    hy = width / ny
    hz = height / nz

    nnx = nx + 1
    nny = ny + 1
    nnz = nz + 1
    total_nodes = nnx * nny * nnz

    nodes = Vector{Node{3,Float64}}(undef, total_nodes)
    
    # Boundary node sets
    left = Int[]
    right = Int[]
    bottom = Int[]
    top = Int[]
    front = Int[]
    back = Int[]

    for k in 0:nz
        z = k * hz
        for j in 0:ny
            y = j * hy
            for i in 0:nx
                x = i * hx
                nid = k * (nnx * nny) + j * nnx + i + 1
                coords = SVector{3,Float64}(x, y, z)
                dofs = [3 * (nid - 1) + 1, 3 * (nid - 1) + 2, 3 * (nid - 1) + 3]
                nodes[nid] = Node{3,Float64}(nid, coords, dofs)

                # Boundary identification
                if i == 0
                    push!(left, nid)
                elseif i == nx
                    push!(right, nid)
                end

                if j == 0
                    push!(bottom, nid)
                elseif j == ny
                    push!(top, nid)
                end

                if k == 0
                    push!(front, nid)
                elseif k == nz
                    push!(back, nid)
                end
            end
        end
    end

    num_elements = nx * ny * nz
    elements = Vector{Element{:Hex8,Float64}}(undef, num_elements)

    eid = 1
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                # Node numbering for hexahedral element
                # Bottom face (z = 0)
                n1 = (k-1) * (nnx * nny) + (j-1) * nnx + i
                n2 = (k-1) * (nnx * nny) + (j-1) * nnx + i + 1
                n3 = (k-1) * (nnx * nny) + j * nnx + i + 1
                n4 = (k-1) * (nnx * nny) + j * nnx + i
                
                # Top face (z = 1)
                n5 = k * (nnx * nny) + (j-1) * nnx + i
                n6 = k * (nnx * nny) + (j-1) * nnx + i + 1
                n7 = k * (nnx * nny) + j * nnx + i + 1
                n8 = k * (nnx * nny) + j * nnx + i

                elements[eid] = Hex8{Float64}(eid, SVector{8,Int}(n1, n2, n3, n4, n5, n6, n7, n8), 1)
                eid += 1
            end
        end
    end

    boundary_nodes = Dict(
        "left" => left,
        "right" => right,
        "bottom" => bottom,
        "top" => top,
        "front" => front,
        "back" => back
    )

    return Mesh{3,Float64,:Hex8}(nodes, elements, boundary_nodes)
end

"""
generate_tetrahedral_mesh(width, height, depth, nx, ny, nz)

Generate structured tetrahedral mesh by subdividing hexahedra into 5 tets each.
Each hexahedral cell is split into 5 tetrahedra following a consistent pattern.
"""
function generate_tetrahedral_mesh(
    width::Real,
    height::Real,
    depth::Real,
    nx::Int,
    ny::Int,
    nz::Int
)
    nx > 0 && ny > 0 && nz > 0 || error("nx, ny, nz must be positive")
    
    hx = width / nx
    hy = height / ny
    hz = depth / nz
    
    nnx = nx + 1
    nny = ny + 1
    nnz = nz + 1
    total_nodes = nnx * nny * nnz
    
    # Generate nodes
    nodes = Vector{Node{3,Float64}}(undef, total_nodes)
    left, right, bottom, top, front, back = Int[], Int[], Int[], Int[], Int[], Int[]
    
    for k in 0:nz
        z = k * hz
        for j in 0:ny
            y = j * hy
            for i in 0:nx
                x = i * hx
                nid = k * (nnx * nny) + j * nnx + i + 1
                coords = SVector{3,Float64}(x, y, z)
                dofs = [3 * (nid - 1) + 1, 3 * (nid - 1) + 2, 3 * (nid - 1) + 3]
                nodes[nid] = Node{3,Float64}(nid, coords, dofs)
                
                if i == 0
                    push!(left, nid)
                elseif i == nx
                    push!(right, nid)
                end
                
                if j == 0
                    push!(bottom, nid)
                elseif j == ny
                    push!(top, nid)
                end
                
                if k == 0
                    push!(front, nid)
                elseif k == nz
                    push!(back, nid)
                end
            end
        end
    end
    
    # Generate tetrahedral elements (5 tets per hex)
    num_elements = nx * ny * nz * 5
    elements = Vector{Element{:Tet4,Float64}}(undef, num_elements)
    
    eid = 1
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                # Hex node numbering
                n1 = (k-1) * (nnx * nny) + (j-1) * nnx + i
                n2 = (k-1) * (nnx * nny) + (j-1) * nnx + i + 1
                n3 = (k-1) * (nnx * nny) + j * nnx + i + 1
                n4 = (k-1) * (nnx * nny) + j * nnx + i
                n5 = k * (nnx * nny) + (j-1) * nnx + i
                n6 = k * (nnx * nny) + (j-1) * nnx + i + 1
                n7 = k * (nnx * nny) + j * nnx + i + 1
                n8 = k * (nnx * nny) + j * nnx + i
                
                # Split hex into 5 tets (standard subdivision pattern)
                elements[eid] = Tet4{Float64}(eid, SVector{4,Int}(n1, n2, n4, n5), 1)
                eid += 1
                elements[eid] = Tet4{Float64}(eid, SVector{4,Int}(n2, n3, n4, n7), 1)
                eid += 1
                elements[eid] = Tet4{Float64}(eid, SVector{4,Int}(n2, n5, n6, n7), 1)
                eid += 1
                elements[eid] = Tet4{Float64}(eid, SVector{4,Int}(n4, n5, n7, n8), 1)
                eid += 1
                elements[eid] = Tet4{Float64}(eid, SVector{4,Int}(n2, n4, n5, n7), 1)
                eid += 1
            end
        end
    end
    
    boundary_nodes = Dict(
        "left" => left,
        "right" => right,
        "bottom" => bottom,
        "top" => top,
        "front" => front,
        "back" => back
    )
    
    return Mesh{3,Float64,:Tet4}(nodes, elements, boundary_nodes)
end

# =============================================================================
# Dimension-aware utility functions
# =============================================================================

"""
get_dofs_per_node(::Type{Mesh{dim,T,ElementType}}) where {dim,T,ElementType}

Get number of DOFs per node based on mesh dimension.
"""
get_dofs_per_node(::Type{Mesh{2,T,ElementType}}) where {T,ElementType} = 2
get_dofs_per_node(::Type{Mesh{3,T,ElementType}}) where {T,ElementType} = 3

"""
get_dofs_per_node(mesh::Mesh)

Get number of DOFs per node for a mesh instance.
"""
get_dofs_per_node(mesh::Mesh{dim,T,ElementType}) where {dim,T,ElementType} = dim

"""
compute_global_dof(node_id::Int, local_dof::Int, mesh::Mesh)

Compute global DOF index from node ID and local DOF.
"""
function compute_global_dof(node_id::Int, local_dof::Int, mesh::Mesh)
    dofs_per_node = get_dofs_per_node(mesh)
    return dofs_per_node * (node_id - 1) + local_dof
end

"""
get_node_local_dofs(mesh::Mesh, node_id::Int)

Get local DOF indices (1, 2, or 1, 2, 3) for a node.
"""
function get_node_local_dofs(mesh::Mesh, node_id::Int)
    dofs_per_node = get_dofs_per_node(mesh)
    return collect(1:dofs_per_node)
end

"""
get_element_coords(mesh::Mesh, element)

Extract coordinates matrix for element nodes.
Returns a dim × n_nodes matrix where each column is a node's coordinates.
"""
function get_element_coords(mesh::Mesh{dim,T}, element) where {dim,T}
    n_nodes = length(element.nodes)
    coords = zeros(T, dim, n_nodes)
    for (local_id, node_id) in enumerate(element.nodes)
        coords[:, local_id] = mesh.nodes[node_id].coords
    end
    return coords
end