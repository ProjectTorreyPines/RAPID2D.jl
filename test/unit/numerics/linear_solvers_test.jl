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

@testitem "LinearSolvers SparseLU pattern change recovery" begin
    # Broadcast-assembled θ-matrices can GROW their pattern when a physical value
    # turns on (sparse broadcast drops numerical zeros). factorize! must recover
    # with a fresh symbolic analysis instead of erroring the step.
    using RAPID2D.SparseArrays
    using RAPID2D: SparseLUSolver, factorize!, solve!

    n = 200
    A1 = spdiagm(0 => fill(3.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
    A2 = A1 + spdiagm(5 => fill(-0.5, n - 5), -5 => fill(-0.5, n - 5))   # pattern grew
    b = [sin(0.1i) + 2.0 for i in 1:n]
    x = zeros(n)

    s = SparseLUSolver{Float64}()
    solve!(x, factorize!(s, A1), b)
    @test maximum(abs, x .- Matrix(A1) \ b) < 1e-12 * maximum(abs, x)
    solve!(x, factorize!(s, A2), b)                    # different pattern → recovery path
    @test maximum(abs, x .- Matrix(A2) \ b) < 1e-12 * maximum(abs, x)
end

@testitem "LinearSolvers electron continuity equivalence" begin
    # The cached-solver path must reproduce `A_LHS \ RHS` exactly. After the solve
    # call, op.A_LHS/op.RHS still hold the assembled system of that step, so the
    # reference solution can be recomputed directly. Three steps exercise both the
    # first-factorization (lu) and the symbolic-reuse (lu!) paths on live matrices.
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 40, NZ = 80,
        prefilled_gas_pressure = 1.0e-2,
        R0B0 = 1.0,
        dt = 1.0e-8,
        snap0D_Δt_s = 1.0e-7,
        snap2D_Δt_s = 1.0e-6,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    for step in 1:3
        RAPID2D.solve_electron_continuity_equation!(RP)
        ne_ref = reshape(RP.operators.A_LHS.matrix \ vec(RP.operators.RHS),
                         RP.G.NR, RP.G.NZ)
        @test isapprox(RP.plasma.ne, ne_ref; rtol = 1e-12)
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
