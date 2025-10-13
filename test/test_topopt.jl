# Concise test suite for topology optimization with ADMM

using Test
using StaticArrays
using LinearAlgebra

using AlgorithmEngineering

@testset "Topology Optimization" begin
    
    # ========================================================================
    # Helper: Create simple test problem (2x2 quad mesh)
    # ========================================================================
    function create_test_problem()
        nodes = [
            FEM.Node{2,Float64}(1, SVector{2,Float64}(0.0, 0.0), [1, 2]),
            FEM.Node{2,Float64}(2, SVector{2,Float64}(1.0, 0.0), [3, 4]),
            FEM.Node{2,Float64}(3, SVector{2,Float64}(2.0, 0.0), [5, 6]),
            FEM.Node{2,Float64}(4, SVector{2,Float64}(0.0, 1.0), [7, 8]),
            FEM.Node{2,Float64}(5, SVector{2,Float64}(1.0, 1.0), [9, 10]),
            FEM.Node{2,Float64}(6, SVector{2,Float64}(2.0, 1.0), [11, 12])
        ]
        
        # Must pre-allocate with abstract Element type due to type invariance
        elements = Vector{FEM.Element{:Quad4,Float64}}(undef, 2)
        elements[1] = FEM.Quad4{Float64}(1, SVector{4,Int}(1, 2, 5, 4), 1)
        elements[2] = FEM.Quad4{Float64}(2, SVector{4,Int}(2, 3, 6, 5), 1)
        
        boundary_nodes = Dict{String, Vector{Int}}()
        mesh = FEM.Mesh{2,Float64,:Quad4}(nodes, elements, boundary_nodes)
        material = FEM.LinearElastic(1.0, 0.3, 1.0)
        
        # Forces: downward on top right node
        forces = Dict(6 => SVector(0.0, -1.0))
        
        # Boundary: fix bottom left corner
        boundary_dofs = [1, 2, 7]  # node 1, both DOFs
        
        problem = ADMM.TopOptProblem(
            mesh = mesh,
            material = material,
            forces = forces,
            boundary_dofs = boundary_dofs,
            vol_frac = 0.5,
            σ_lim = 1.0,
            r_filter = 1.5,
            β_heaviside = 1.0,
            η_heaviside = 0.5,
            max_iter_mma = 10,
            mma_tol = 1e-2
        )
        
        return problem
    end
    
    # ========================================================================
    # Test 1: Problem Setup
    # ========================================================================
    @testset "Setup" begin
        problem = create_test_problem()
        state = ADMM.init(problem)
        
        @test state.n == 2  # 2 elements
        @test length(state.ctx.ϕ) == 2
        @test length(state.ctx.ρ) == 2
        @test length(state.ctx.α) == 2
        @test length(state.ctx.λ) == 2
        @test size(state.ctx.σ) == (3, 2)  # 3 stress components × 2 elements
        @test length(state.ctx.U) == 12  # 6 nodes × 2 DOF
        @test state.iter == 0
    end
    
    # ========================================================================
    # Test 2: Density Filtering
    # ========================================================================
    @testset "Filtering" begin
        problem = create_test_problem()
        state = ADMM.init(problem)
        
        # Set design variables
        state.ctx.ϕ .= [0.3, 0.7]
        
        ADMM.apply_density_filter!(state)
        
        @test all(state.ctx.ρ .>= problem.ρ_min)
        @test all(state.ctx.ρ .<= problem.ρ_max)
        @test length(state.ctx.ρ) == 2
    end
    
    # ========================================================================
    # Test 3: Stress Computation
    # ========================================================================
    @testset "Stress Computation" begin
        problem = create_test_problem()
        state = ADMM.init(problem)
        
        # Apply filter and build stiffness
        state.ctx.ϕ .= [0.5, 0.5]
        ADMM.apply_density_filter!(state)
        
        state.ctx.K = FEM.assemble_stiffness_matrix(
            problem.mesh, problem.material, problem.analysis_type;
            ρ = state.ctx.ρ, ρ_simp = problem.ρ_simp,
            E_min = problem.ρ_min * problem.material.E
        )
        
        state.ctx.U = FEM.solve_fem(state.ctx.K, state.ctx.f, problem.boundary_dofs)
        ADMM.compute_element_stresses!(state)
        
        @test all(state.ctx.σ̄ .>= 0)  # Von Mises stress always positive
        @test all(state.ctx.σ̃ .>= 0)  # Relaxed stress positive
        @test length(state.ctx.σ̄) == 2
        @test length(state.ctx.σ̃) == 2
    end
    
    # ========================================================================
    # Test 4: Gradient Computation
    # ========================================================================
    @testset "Adjoint Gradients" begin
        problem = create_test_problem()
        state = ADMM.init(problem)
        
        # Full forward pass
        state.ctx.ϕ .= [0.5, 0.5]
        ADMM.apply_density_filter!(state)
        
        state.ctx.K = FEM.assemble_stiffness_matrix(
            problem.mesh, problem.material, problem.analysis_type;
            ρ = state.ctx.ρ, ρ_simp = problem.ρ_simp,
            E_min = problem.ρ_min * problem.material.E
        )
        
        state.ctx.U = FEM.solve_fem(state.ctx.K, state.ctx.f, problem.boundary_dofs)
        ADMM.compute_element_stresses!(state)
        
        # Initialize ADMM variables
        state.ctx.α .= state.ctx.σ̃
        state.ctx.λ .= 0.0
        
        # Compute gradients
        ADMM.compute_gradients_adjoint!(state)
        
        @test length(state.ctx.∇L_ϕ) == 2
        @test !all(state.ctx.∇L_ϕ .== 0)  # Should be non-zero
        @test all(isfinite.(state.ctx.∇L_ϕ))  # Should be finite
    end
    
    # ========================================================================
    # Test 5: Single ADMM Step
    # ========================================================================
    @testset "ADMM Step" begin
        problem = create_test_problem()
        state = ADMM.init(problem, params=ADMM.ADMMParams(μ=1.0, abstol=1e-2, reltol=1e-2))
        
        # Run one step
        converged, r_p, r_d, ϵ_p, ϵ_d = ADMM._step!(state)
        
        @test state.iter == 1
        @test isfinite(r_p)
        @test isfinite(r_d)
        @test !all(state.ctx.ϕ .== 0.5)  # Design should have changed
    end
    
    # ========================================================================
    # Test 6: Convergence (quick)
    # ========================================================================
    @testset "Convergence" begin
        problem = create_test_problem()
        state = ADMM.init(problem, params=ADMM.ADMMParams(
            μ=1.0, 
            abstol=0.1,  # Loose tolerances for speed
            reltol=0.1,
            adaptive_μ=false
        ))
        
        # Run a few iterations (not to full convergence, just check it runs)
        state_out, iters, converged = ADMM.run_admm!(state, max_iter=5, verbose=false)
        
        @test iters <= 5
        @test state_out.iter == iters
        @test all(state_out.ctx.ρ .>= problem.ρ_min)
        @test all(state_out.ctx.ρ .<= problem.ρ_max)
    end
    
    # ========================================================================
    # Test 7: Adaptive Parameters
    # ========================================================================
    @testset "Adaptive Parameters" begin
        problem = create_test_problem()
        state = ADMM.init(problem, params=ADMM.ADMMParams(μ=1.0, adaptive_μ=true))
        
        μ_initial = state.params.μ
        β_initial = problem.β_heaviside
        
        # Run 5 iterations to trigger adaptive μ (every 5 iters)
        for i in 1:5
            ADMM._step!(state)
        end
        
        @test state.params.μ >= μ_initial  # μ should increase
        
        # Run 45 more iterations to trigger adaptive β (every 50 iters)
        for i in 1:45
            ADMM._step!(state)
        end
        
        @test problem.β_heaviside >= β_initial  # β should increase

        # Grey fraction driven β update should trigger even without schedule
        problem_grey = create_test_problem()
        problem_grey.β_heaviside = 1.0
        problem_grey.β_heaviside_growth = 2.0
        problem_grey.β_heaviside_max = 4.0
        problem_grey.β_update_frequency = typemax(Int) # disable periodic growth
        problem_grey.grey_fraction_trigger = 0.1

        state_grey = ADMM.init(problem_grey)
        state_grey.iter = 1
        state_grey.ctx.ρ .= 0.5 # intentionally grey distribution

        ADMM.update_heaviside_sharpness!(state_grey)

        @test problem_grey.β_heaviside > 1.0
    end
    
end

println("✓ All topology optimization tests passed!")
