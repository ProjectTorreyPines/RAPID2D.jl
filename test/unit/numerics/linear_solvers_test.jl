# Cached-factorization linear solvers (SparseLUSolver / BandedLUSolver): pure
# sparse-matrix tests, no RAPID object. Contract: factorize!(s, A) after A's
# values change (pattern must stay fixed for the symbolic-reuse path), then
# solve!(X, s, B) with vector or matrix RHS.

@testitem "LinearSolvers correctness and refactorization" begin
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra
    using RAPID2D: AbstractLinearSolver, SparseLUSolver, BandedLUSolver, factorize!, solve!

    n = 400
    A = spdiagm(0 => fill(4.0, n), 1 => fill(-0.9, n - 1), -1 => fill(-0.9, n - 1),
                20 => fill(-0.8, n - 20), -20 => fill(-0.8, n - 20))
    B = [sin(0.01i + j) + 2.0 for i in 1:n, j in 1:6]
    Xref = Matrix(A) \ B

    for s in (SparseLUSolver{Float64}(), BandedLUSolver(A))
        @test s isa AbstractLinearSolver{Float64}
        X = similar(B)
        solve!(X, factorize!(s, A), B)
        @test maximum(abs, X .- Xref) / maximum(abs, Xref) < 1e-12

        # same pattern, new values → symbolic-reuse refactorization path
        A2 = A .* 1.5
        solve!(X, factorize!(s, A2), B)
        @test maximum(abs, X .- Matrix(A2) \ B) / maximum(abs, Xref) < 1e-12

        # vector RHS path
        x = zeros(n)
        solve!(x, s, B[:, 1])
        @test maximum(abs, x .- Matrix(A2) \ B[:, 1]) / maximum(abs, Xref) < 1e-12
    end
end

@testitem "LinearSolvers BandedLU residual fallback" begin
    using RAPID2D.SparseArrays
    using RAPID2D: BandedLUSolver, factorize!, solve!

    n = 300
    A = spdiagm(0 => fill(3.0, n), 2 => fill(-1.0, n - 2), -2 => fill(-1.0, n - 2))
    B = [cos(0.02i) + 1.5 + j for i in 1:n, j in 1:3]
    Xref = Matrix(A) \ B

    # unreachable tolerance → the residual check must trigger the SparseLU fallback,
    # which still returns the correct solution
    s = BandedLUSolver(A; resid_tol=0.0)
    X = similar(B)
    solve!(X, factorize!(s, A), B)
    @test maximum(abs, X .- Xref) / maximum(abs, Xref) < 1e-12
    @test s.fallback.F !== nothing   # fallback actually engaged
end
