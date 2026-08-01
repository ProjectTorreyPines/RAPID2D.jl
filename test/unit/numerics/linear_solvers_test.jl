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

@testitem "LinearSolvers Te theta-scheme equivalence" begin
    # This commit routes BOTH solves through the cached factorization, but only
    # the electron-continuity side had an equivalence test. `update_Te!` uses its
    # own cached `OP.Te_solver` (physics.jl), so it needs its own check.
    #
    # Same contract as the continuity test: after the call, op.A_LHS/op.RHS still
    # hold the assembled system of that step, so the reference can be recomputed
    # directly. One difference — `update_Te!` clamps to [min_Te, max_Te] AFTER
    # the solve, so the reference must be clamped too, or the comparison is
    # against a quantity the function never claims to produce.
    #
    # Te MUST be given a non-uniform profile. A_LHS = II minus the dt-weighted
    # transport operators, and a uniform field lies in the kernel of every one of
    # them: A*(const) == (const) no matter how large dt is. Measured, with a flat
    # Te the direct solve differs from RHS by 5e-16 at dt=1e-6 AND by 8e-16 at
    # dt=1e-4 (where |A-I| is already 1e-2) — i.e. a solver that ignored the
    # matrix entirely would pass. Raising dt does not fix this; breaking the
    # symmetry does. With the Gaussian below the gap is 1e-2, ten orders above
    # the tolerance. The final assertion pins that discriminating power so this
    # test can never silently decay into a tautology.
    using RAPID2D.SparseArrays
    using RAPID2D.LinearAlgebra

    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 40, NZ = 80,
        prefilled_gas_pressure = 1.0e-2,
        R0B0 = 1.0,
        dt = 1.0e-5,
        snap0D_Δt_s = 1.0,
        snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # the cached path is only reached through the implicit branch, and A_LHS is
    # only non-trivial when at least one transport term is on
    @test RP.flags.Implicit
    @test RP.flags.Include_Te_diffu_term || RP.flags.Include_Te_convec_term
    @test !RP.flags.evolve_Te_inWall_only        # the other branch is a stub

    Rc = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    Zc = (RP.G.Z1D[1] + RP.G.Z1D[end]) / 2
    @. RP.plasma.Te_eV = 5.0 +
        4.0 * exp(-((RP.G.R2D - Rc)^2 + (RP.G.Z2D - Zc)^2) / 0.02)
    RAPID2D.update_transport_quantities!(RP)

    for step in 1:3
        RAPID2D.update_Te!(RP)
        Te_ref = reshape(RP.operators.A_LHS.matrix \ vec(RP.operators.RHS),
                         RP.G.NR, RP.G.NZ)
        clamp!(Te_ref, config.min_Te, config.max_Te)
        @test isapprox(RP.plasma.Te_eV, Te_ref; rtol = 1e-12)

        # discriminating power: how far the true solution sits from the answer a
        # matrix-ignoring solver would return. Must stay far above the rtol above.
        naive = clamp.(RP.operators.RHS, config.min_Te, config.max_Te)
        @test maximum(abs, naive .- Te_ref) / maximum(abs, Te_ref) > 1.0e-6
    end
end
