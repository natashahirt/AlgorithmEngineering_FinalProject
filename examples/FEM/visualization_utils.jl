using GLMakie
using AlgorithmEngineering.FEM
using Colors
using GeometryBasics

const HEX8_EDGE_PAIRS = (
    (1, 2), (2, 3), (3, 4), (4, 1),     # Bottom face
    (5, 6), (6, 7), (7, 8), (8, 5),     # Top face
    (1, 5), (2, 6), (3, 7), (4, 8)      # Vertical edges
)

# Face definitions for Hex8 element (counterclockwise when viewed from outside)
const HEX8_FACES = (
    (1, 4, 3, 2),  # Bottom face (z = z_min)
    (5, 6, 7, 8),  # Top face (z = z_max)
    (1, 2, 6, 5),  # Front face (y = y_min)
    (4, 8, 7, 3),  # Back face (y = y_max)
    (1, 5, 8, 4),  # Left face (x = x_min)
    (2, 3, 7, 6)   # Right face (x = x_max)
)

"""
    compute_fem_visualization_data(mesh, displacements; scale=1.0)

Prepare common data structures for FEM deformation visualizations.
"""
function compute_fem_visualization_data(mesh, displacements; scale=1.0)
    nodes = FEM.get_nodes(mesh)
    elements = FEM.get_elements(mesh)
    n_nodes = length(nodes)

    x_orig = [node.coords[1] for node in nodes]
    y_orig = [node.coords[2] for node in nodes]

    x_deformed = similar(x_orig)
    y_deformed = similar(y_orig)
    u_magnitude = similar(x_orig)

    for (i, node) in enumerate(nodes)
        dof_x = node.dofs[1]
        dof_y = node.dofs[2]

        u_x = displacements[dof_x]
        u_y = displacements[dof_y]

        x_deformed[i] = x_orig[i] + scale * u_x
        y_deformed[i] = y_orig[i] + scale * u_y
        u_magnitude[i] = hypot(u_x, u_y)
    end

    return (; nodes, elements, x_orig, y_orig, x_deformed, y_deformed, u_magnitude, scale)
end

"""
    compute_fem_visualization_data_3d(mesh, displacements; scale=1.0)

Prepare common data structures for 3D FEM deformation visualizations.
"""
function compute_fem_visualization_data_3d(mesh::FEM.Mesh{3}, displacements; scale=1.0)
    nodes = FEM.get_nodes(mesh)
    elements = FEM.get_elements(mesh)
    n_nodes = length(nodes)

    x_orig = Vector{Float64}(undef, n_nodes)
    y_orig = Vector{Float64}(undef, n_nodes)
    z_orig = Vector{Float64}(undef, n_nodes)

    x_deformed = similar(x_orig)
    y_deformed = similar(y_orig)
    z_deformed = similar(z_orig)
    u_magnitude = similar(x_orig)

    for (i, node) in enumerate(nodes)
        coords = node.coords
        dofs = node.dofs

        u_x = displacements[dofs[1]]
        u_y = displacements[dofs[2]]
        u_z = displacements[dofs[3]]

        x_orig[i] = coords[1]
        y_orig[i] = coords[2]
        z_orig[i] = coords[3]

        x_deformed[i] = x_orig[i] + scale * u_x
        y_deformed[i] = y_orig[i] + scale * u_y
        z_deformed[i] = z_orig[i] + scale * u_z
        u_magnitude[i] = sqrt(u_x^2 + u_y^2 + u_z^2)
    end

    return (; nodes, elements,
            x_orig, y_orig, z_orig,
            x_deformed, y_deformed, z_deformed,
            u_magnitude, scale)
end

"""
    create_fem_deformation_figure(data; kwargs...)

Generate a side-by-side figure showing undeformed and deformed FEM meshes.
"""
function create_fem_deformation_figure(data; figure_size=(1400, 600), title="FEM Analysis",
                                        colormap=:viridis, colorrange=nothing, show_nodes=true,
                                        overlay_original=false, annotate_undeformed=nothing,
                                        annotate_deformed=nothing,
                                        colorbar_label="Displacement Magnitude [m]")
    limits = isnothing(colorrange) ? (0.0, maximum(data.u_magnitude)) : colorrange

    fig = Figure(size=figure_size)

    ax1 = Axis(fig[1, 1],
               aspect=DataAspect(),
               title="Undeformed Mesh",
               xlabel="x [m]",
               ylabel="y [m]")

    for elem in data.elements
        node_ids = collect(elem.nodes)
        x_elem = [data.x_orig[id] for id in node_ids]
        y_elem = [data.y_orig[id] for id in node_ids]
        push!(x_elem, x_elem[1])
        push!(y_elem, y_elem[1])
        lines!(ax1, x_elem, y_elem, color=:black, linewidth=1)
    end

    if show_nodes
        scatter!(ax1, data.x_orig, data.y_orig, color=:blue, markersize=3)
    end

    if annotate_undeformed !== nothing
        annotate_undeformed(ax1, data)
    end

    ax2 = Axis(fig[1, 2],
               aspect=DataAspect(),
               title="Deformed Shape (scale = $(round(data.scale, digits=1)))",
               xlabel="x [m]",
               ylabel="y [m]")

    for elem in data.elements
        node_ids = collect(elem.nodes)
        x_elem = [data.x_deformed[id] for id in node_ids]
        y_elem = [data.y_deformed[id] for id in node_ids]
        u_elem = [data.u_magnitude[id] for id in node_ids]

        u_avg = sum(u_elem) / length(u_elem)

        poly!(ax2, Point2f.(x_elem, y_elem),
              color=u_avg, colormap=colormap, colorrange=limits)
    end

    for elem in data.elements
        node_ids = collect(elem.nodes)
        x_elem = [data.x_deformed[id] for id in node_ids]
        y_elem = [data.y_deformed[id] for id in node_ids]
        push!(x_elem, x_elem[1])
        push!(y_elem, y_elem[1])

        lines!(ax2, x_elem, y_elem, color=:black, linewidth=0.5, alpha=0.3)
    end

    if overlay_original
        for elem in data.elements
            node_ids = collect(elem.nodes)
            x_elem = [data.x_orig[id] for id in node_ids]
            y_elem = [data.y_orig[id] for id in node_ids]
            push!(x_elem, x_elem[1])
            push!(y_elem, y_elem[1])
            lines!(ax2, x_elem, y_elem, color=(:gray, 0.3), linewidth=0.5, linestyle=:dash)
        end
    end

    if annotate_deformed !== nothing
        annotate_deformed(ax2, data)
    end

    Colorbar(fig[1, 3],
             limits=limits,
             colormap=colormap,
             label=colorbar_label)

    Label(fig[0, :], title, fontsize=20, font=:bold)

    return fig
end

"""
    create_fem_deformation_figure_3d(data; kwargs...)

Create a side-by-side 3D visualization comparing undeformed and deformed meshes.
"""
function create_fem_deformation_figure_3d(data;
        figure_size=(1400, 650),
        title="FEM 3D Analysis",
        colormap=:viridis,
        colorrange=nothing,
        show_nodes=true,
        overlay_original=true,
        edge_color=RGBA(0.15, 0.15, 0.15, 0.8),
        edge_width=0.8,
        colorbar_label="Displacement Magnitude [m]",
        load_node=nothing)

    u_max = maximum(data.u_magnitude)
    default_upper = u_max == 0.0 ? 1.0 : u_max
    limits = isnothing(colorrange) ? (0.0, default_upper) : colorrange

    fig = Figure(size=figure_size)

    ax1 = Axis3(fig[1, 1],
                aspect=:data,
                title="Undeformed Mesh",
                xlabel="x [m]",
                ylabel="y [m]",
                zlabel="z [m]")

    # Draw edges for undeformed mesh (wireframe)
    _draw_hex8_edges!(ax1, data.elements,
                      data.x_orig, data.y_orig, data.z_orig;
                      color=:black, linewidth=1.0)
    
    # Draw nodes for undeformed mesh
    if show_nodes
        scatter!(ax1,
                 Point3f.(data.x_orig, data.y_orig, data.z_orig),
                 markersize=4,
                 color=:blue)
    end
    
    # Mark load point with red dot
    if !isnothing(load_node)
        scatter!(ax1,
                 [Point3f(data.x_orig[load_node], data.y_orig[load_node], data.z_orig[load_node])],
                 markersize=15,
                 color=:red,
                 marker=:circle)
    end

    ax2 = Axis3(fig[1, 2],
                aspect=:data,
                title="Deformed Shape (scale = $(round(data.scale, digits=1)))",
                xlabel="x [m]",
                ylabel="y [m]",
                zlabel="z [m]")

    # Draw colored faces for deformed mesh
    _draw_hex8_faces!(ax2, data.elements,
                      data.x_deformed, data.y_deformed, data.z_deformed, data.u_magnitude;
                      colormap=colormap, colorrange=limits)
    
    # Draw thin black edges on top of colored faces
    _draw_hex8_edges!(ax2, data.elements,
                      data.x_deformed, data.y_deformed, data.z_deformed;
                      color=:black, linewidth=0.5)
    
    # Mark load point with red dot on deformed mesh
    if !isnothing(load_node)
        scatter!(ax2,
                 [Point3f(data.x_deformed[load_node], data.y_deformed[load_node], data.z_deformed[load_node])],
                 markersize=15,
                 color=:red,
                 marker=:circle)
    end

    # Create colorbar
    Colorbar(fig[1, 3], 
             limits=limits,
             colormap=colormap,
             label=colorbar_label)
    Label(fig[0, :], title, fontsize=20, font=:bold)

    return fig
end

"""
    visualize_fem_deformation_3d(mesh, displacements; kwargs...)

Convenience wrapper to generate a 3D deformation visualization directly from
the mesh and displacement vector.
"""
function visualize_fem_deformation_3d(mesh::FEM.Mesh{3}, displacements; scale=1.0, kwargs...)
    data = compute_fem_visualization_data_3d(mesh, displacements; scale=scale)
    return create_fem_deformation_figure_3d(data; kwargs...)
end

function _draw_hex8_edges!(ax, elements, x, y, z; color, linewidth)
    for element in elements
        node_ids = Tuple(element.nodes)
        for (a, b) in HEX8_EDGE_PAIRS
            pa = Point3f(x[node_ids[a]], y[node_ids[a]], z[node_ids[a]])
            pb = Point3f(x[node_ids[b]], y[node_ids[b]], z[node_ids[b]])
            linesegments!(ax, [pa, pb]; color=color, linewidth=linewidth)
        end
    end
end

function _draw_hex8_faces!(ax, elements, x, y, z, u_magnitude; colormap, colorrange)
    # Build a map of faces to track which are on the exterior
    face_map = Dict{Set{Int}, Tuple{Vector{Point3f}, Float64}}()
    
    for element in elements
        node_ids = Tuple(element.nodes)
        
        # Compute average displacement for this element
        u_elem = [u_magnitude[nid] for nid in node_ids]
        u_avg = sum(u_elem) / length(u_elem)
        
        # Process each face
        for face_nodes in HEX8_FACES
            # Create a set of global node IDs for this face
            face_key = Set(node_ids[i] for i in face_nodes)
            points = [Point3f(x[node_ids[i]], y[node_ids[i]], z[node_ids[i]]) for i in face_nodes]
            
            # If face already exists, it's interior (shared by two elements) - remove it
            # If face is new, it's on the exterior - add it
            if haskey(face_map, face_key)
                delete!(face_map, face_key)
            else
                face_map[face_key] = (points, u_avg)
            end
        end
    end
    
    # Draw only exterior faces
    for (points, u_avg) in values(face_map)
        mesh_obj = GeometryBasics.Mesh(points, [GeometryBasics.QuadFace(1, 2, 3, 4)])
        mesh!(ax, mesh_obj; 
              color=u_avg, 
              colormap=colormap, 
              colorrange=colorrange)
    end
end
