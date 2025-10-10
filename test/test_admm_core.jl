using Test
using ADMM
using LinearAlgebra

@testset "ADMM Core Tests" begin
    
    @testset "Basic Initialization" begin
        m, n = 20, 10
        A = randn(m, n)
        b = randn(m)
        
        problem = LassoProblem(A=A, b=b, λ=0.1, distribution=ADMM.Serial())
        state = init(problem; params=ADMMParams())
        
        @test state.m == m
        @test state.n == n
        @test all(state.x .== 0.0)
    end
    
    @testset "Convergence" begin
        # Solve Ax ≈ b with known solution
        m, n = 30, 15
        A = randn(m, n)
        x_true = randn(n)
        b = A * x_true
        
        problem = LassoProblem(A=A, b=b, λ=0.001, distribution=ADMM.Serial())
        state = init(problem; params=ADMMParams(reltol=1e-3, abstol=1e-4))
        state, iters, converged = run_admm!(state; max_iter=500, verbose=false)
        
        @test converged == true
        @test norm(state.z - x_true) / norm(x_true) < 0.1
    end
    
    @testset "Objective Evaluation" begin
        A = randn(10, 5)
        b = randn(10)
        x = randn(5)
        problem = LassoProblem(A=A, b=b, λ=0.5)
        
        @test evaluate_objective(problem, x) ≈ 0.5 * norm(A * x - b)^2
        @test evaluate_global_regularizer(problem, x) ≈ 0.5 * norm(x, 1)
    end
end