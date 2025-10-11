using Test
using StaticArrays
using SparseArrays
using ADMM
using ADMM.FEM

@testset "FEM 2D interface" begin
    mesh = generate_rectangular_mesh(1.0, 1.0, 1, 1)
    material = LinearElastic(200.0, 0.3)
    analysis = PlaneStress()

    @test get_dofs_per_node(mesh) == 2
    @test length(get_element_dofs(mesh, mesh.elements[1])) == 8

    ndof = maximum(last(node.dofs) for node in mesh.nodes)
    displacements = zeros(Float64, ndof)

    # Impose a linear displacement field ux = x, uy = 0 → εxx = 1
    for node in mesh.nodes
        x = node.coords[1]
        displacements[node.dofs[1]] = x
        displacements[node.dofs[2]] = 0.0
    end

    σ = compute_element_stresses(mesh, material, displacements, 1, analysis)
    strain = @SVector [1.0, 0.0, 0.0]
    expected = get_constitutive_matrix(material, analysis) * strain

    @test σ ≈ expected atol = 1e-10

    nodal = compute_nodal_stresses(mesh, material, displacements, analysis)
    for row in eachrow(nodal)
        @test collect(row) ≈ collect(expected) atol = 1e-10
    end

    left_nodes = get_boundary_nodes(mesh, "left")
    fixed_bcs = create_fixed_support(mesh, left_nodes)
    @test length(fixed_bcs) == 2 * length(left_nodes)

    dofs, values = collect_dirichlet_constraints(mesh, fixed_bcs)
    @test all(iszero, values)
    expected_dofs = Int[]
    for node in mesh.nodes
        if node.id in left_nodes
            append!(expected_dofs, node.dofs)
        end
    end
    @test sort(dofs) == sort(expected_dofs)

    K = assemble_stiffness_matrix(mesh, material, analysis)
    f = zeros(Float64, size(K, 1))
    K_bc, f_bc = apply_boundary_conditions!(K, f, mesh, fixed_bcs)

    for dof in dofs
        row = K_bc[dof, :]
        col = K_bc[:, dof]
        @test isapprox(row[dof], 1.0; atol = 1e-12)
        @test nnz(row) == 1
        @test nnz(col) == 1
        @test iszero(f_bc[dof])
    end

    load_case = LoadCase()
    append!(load_case.dirichlet_bcs, fixed_bcs)
    right_node = last(get_boundary_nodes(mesh, "right"))
    merge!(load_case.point_loads, create_point_load(mesh, right_node, 2, 5.0))

    f_load = zeros(Float64, ndof)
    apply_loads!(f_load, mesh, load_case)
    right_dof = compute_global_dof(right_node, 2, mesh)
    @test f_load[right_dof] ≈ 5.0

    disp_matrix = extract_displacements(mesh, displacements, [1, 2])
    @test size(disp_matrix) == (2, 2)
    disp_vector = extract_displacements(mesh, displacements, [1, 2], 1)
    @test disp_vector == displacements[[compute_global_dof(1, 1, mesh), compute_global_dof(2, 1, mesh)]]
end

@testset "FEM 3D interface" begin
    mesh = generate_hexahedral_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    material = LinearElastic(150.0, 0.25)
    analysis = ThreeDimensional()

    @test get_dofs_per_node(mesh) == 3
    @test length(get_element_dofs(mesh, mesh.elements[1])) == 24

    ndof = maximum(last(node.dofs) for node in mesh.nodes)
    displacements = zeros(Float64, ndof)

    # Impose a linear displacement field ux = x, uy = uz = 0 → εxx = 1
    for node in mesh.nodes
        x = node.coords[1]
        displacements[node.dofs[1]] = x
        displacements[node.dofs[2]] = 0.0
        displacements[node.dofs[3]] = 0.0
    end

    σ = compute_element_stresses(mesh, material, displacements, 1, analysis)
    strain = @SVector [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected = get_constitutive_matrix(material, analysis) * strain

    @test σ ≈ expected atol = 1e-10

    nodal = compute_nodal_stresses(mesh, material, displacements, analysis)
    for row in eachrow(nodal)
        @test collect(row) ≈ collect(expected) atol = 1e-10
    end
end

@testset "Boundary application" begin
    mesh = generate_rectangular_mesh(1.0, 1.0, 1, 1)
    material = LinearElastic(120.0, 0.25)
    analysis = PlaneStress()

    K = assemble_stiffness_matrix(mesh, material, analysis)
    F = zeros(Float64, size(K, 1))

    constrained_dof = 1
    prescribed_value = 0.1
    col_reference = Array(K[:, constrained_dof])

    K_bc, F_bc = apply_boundary_conditions(K, F, [constrained_dof], [prescribed_value])

    other_dofs = setdiff(collect(1:length(F_bc)), [constrained_dof])

    @test isapprox(K_bc[constrained_dof, constrained_dof], 1.0; atol = 1e-12)
    @test all(isapprox.(K_bc[constrained_dof, other_dofs], 0.0; atol = 1e-12))
    @test all(isapprox.(K_bc[other_dofs, constrained_dof], 0.0; atol = 1e-12))
    @test F_bc[constrained_dof] ≈ prescribed_value
    @test all(isapprox.(F_bc[other_dofs], -prescribed_value .* col_reference[other_dofs]; atol = 1e-12))

    dirichlet_bcs = [DirichletBC{Float64}([get_boundary_nodes(mesh, "left")[1]], 1, prescribed_value)]
    K_bc2, F_bc2 = apply_boundary_conditions!(K, F, mesh, dirichlet_bcs)

    other_dofs2 = setdiff(collect(1:length(F_bc2)), [constrained_dof])

    @test isapprox(K_bc2[constrained_dof, constrained_dof], 1.0; atol = 1e-12)
    @test F_bc2[constrained_dof] ≈ prescribed_value
    @test all(isapprox.(F_bc2[other_dofs2], -prescribed_value .* col_reference[other_dofs2]; atol = 1e-12))
end

@testset "Load helpers" begin
    mesh3d = generate_hexahedral_mesh(1.0, 1.0, 1.0, 1, 1, 1)

    neumann = [
        NeumannBC{Float64}([1, 2], 3, 5.0),
        NeumannBC{Float64}([2], 3, -1.0),
    ]

    loads = collect_neumann_loads(mesh3d, neumann)

    dof_node1 = compute_global_dof(1, 3, mesh3d)
    dof_node2 = compute_global_dof(2, 3, mesh3d)

    @test loads[dof_node1] ≈ 5.0
    @test loads[dof_node2] ≈ 4.0

    point_load_float = create_point_load(Float32, mesh3d, 1, 1, 3.5)
    @test eltype(values(point_load_float)) == Float32
    @test point_load_float[compute_global_dof(1, 1, mesh3d)] ≈ Float32(3.5)
end

@testset "Stress utilities" begin
    vm2d = compute_von_mises_stress([100.0, 50.0, 10.0])
    expected_vm2d = sqrt(100.0^2 - 100.0 * 50.0 + 50.0^2 + 3 * 10.0^2)
    @test vm2d ≈ expected_vm2d

    vm3d = compute_von_mises_stress([100.0, 50.0, 25.0, 5.0, 4.0, 3.0])
    expected_vm3d = sqrt(0.5 * ((100.0 - 50.0)^2 + (50.0 - 25.0)^2 + (25.0 - 100.0)^2 + 6 * (5.0^2 + 4.0^2 + 3.0^2)))
    @test vm3d ≈ expected_vm3d

    σ1, σ2, σ3 = compute_principal_stresses([90.0, 30.0, 15.0])
    avg = (90.0 + 30.0) / 2
    τ_max = sqrt(((90.0 - 30.0) / 2)^2 + 15.0^2)
    @test σ1 ≈ avg + τ_max
    @test σ2 ≈ avg - τ_max
    @test σ3 ≈ 0.0

    p1, p2, p3 = compute_principal_stresses([80.0, 40.0, 20.0, 0.0, 0.0, 0.0])
    @test p1 ≈ 80.0
    @test p2 ≈ 40.0
    @test p3 ≈ 20.0

    energy = compute_strain_energy([2.0 0.0; 0.0 4.0], [1.0, 3.0])
    @test energy ≈ 19.0

    compliance = compute_compliance([1.0, 2.0], [3.0, 4.0])
    @test compliance ≈ 11.0
end
