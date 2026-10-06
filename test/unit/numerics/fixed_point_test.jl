# Newton's step for a fixed point x = g(x), and the solve that drives a stepper, apart from any
# physics. On an affine map g(x) = T x + c, Newton's step with the exact Jacobian lands on the
# fixed point at once, whatever the eigenvalues of T, and later steps only remove rounding.

@testsnippet AffineMap begin
    using LinearAlgebra
    # T with eigenvalues −5 (the relaxed iteration diverges) and 0.999 (it stalls) among others,
    # in a fixed orthogonal basis; the unknowns on scales from 1e-6 to 1e6.
    n = 8
    λ = [-5.0, 0.999, 0.5, 0.2, 0.0, -0.3, 0.1, 0.7]
    Q = Matrix(qr([sin(i * j + i) for i in 1:n, j in 1:n]).Q)
    scale = 10.0 .^ range(-6, 6; length = n)
    T = Diagonal(scale) * (Q * Diagonal(λ) * Q') / Diagonal(scale)
    c = scale .* cos.(1:n)
    x_star = (I - T) \ c
    g(x) = T * x + c
    W = 1 ./ scale   # the unknowns in one unit
    newton_lu() = lu(Diagonal(W) * (I - T) / Diagonal(W))
end

@testitem "Newton's step lands on an affine map's fixed point" setup = [AffineMap] begin
    using RAPID2D: NewtonStepper, fixed_point_step!
    S = NewtonStepper(newton_lu(), W)
    x0 = zeros(n)
    x1, status = fixed_point_step!(S, x0, g(x0) .- x0)
    @test status === :best
    @test maximum(abs, W .* (x1 .- x_star)) <= 1.0e-12 * maximum(abs, W .* x_star)
end

@testitem "Newton's step: a residual no smaller than the best stalls" setup = [AffineMap] begin
    using RAPID2D: NewtonStepper, fixed_point_step!
    S = NewtonStepper(newton_lu(), W)
    x0 = ones(n)
    f0 = g(x0) .- x0
    @test fixed_point_step!(S, x0, f0)[2] === :best
    x, status = fixed_point_step!(S, x0, f0)
    @test status === :stalled
    @test x == x0
end

@testitem "Newton's step: a residual that is not finite" setup = [AffineMap] begin
    using RAPID2D: NewtonStepper, fixed_point_step!
    # the first: nothing to fall back on; a later one: the solve ends on the best iterate
    S = NewtonStepper(newton_lu(), W)
    @test fixed_point_step!(S, zeros(n), fill(NaN, n))[2] === :failed
    @test fixed_point_step!(S, zeros(n), g(zeros(n)))[2] === :best
    @test fixed_point_step!(S, ones(n), fill(Inf, n)) == (zeros(n), :exhausted)
    @test fixed_point_step!(S, ones(n), zeros(n); valid = false) == (zeros(n), :exhausted)
end

@testitem "Fixed-point solve with Newton's step: two evaluations on an affine map" setup = [AffineMap] begin
    using RAPID2D: NewtonStepper, fixed_point_solve!
    # The stopping test asks for 1e-10 of the solution in the weighted unknowns: the first
    # evaluation does not meet it, the one after Newton's step does.
    n_eval, kept = Ref(0), Ref(0)
    function evaluate!(x)
        n_eval[] += 1
        f = g(x) .- x
        return f, maximum(abs, W .* f) <= 1.0e-10 * maximum(abs, W .* x_star), true
    end
    iter, outcome = fixed_point_solve!(
        evaluate!, NewtonStepper(newton_lu(), W), zeros(n); max_iter = 20, keep! = () -> (kept[] += 1), restore! = () -> nothing,
    )
    @test (iter, outcome) == (2, :converged)
    @test n_eval[] == 2 && kept[] == 2
end

@testitem "Fixed-point solve with Newton's step: a test no evaluation meets ends on the best" setup = [AffineMap] begin
    using RAPID2D: NewtonStepper, fixed_point_solve!
    # A stopping test that never holds: once only rounding is left the step stalls, and the
    # solve stops there, restoring its best evaluation, long before max_iter.
    restored = Ref(0)
    iter, outcome = fixed_point_solve!(
        x -> (g(x) .- x, false, true), NewtonStepper(newton_lu(), W), zeros(n);
        max_iter = 50, keep! = () -> nothing, restore! = () -> (restored[] += 1),
    )
    @test outcome === :stopped
    @test 2 < iter < 50
    @test restored[] == 1
end

@testitem "Fixed-point solve: a stalled evaluation is accepted only if it meets the test" begin
    using RAPID2D: NewtonStepper, fixed_point_solve!
    using LinearAlgebra
    # A residual that never falls: the second evaluation stalls, and ends the solve. It is
    # accepted if it meets the stopping test; if not, the best (first) evaluation is restored.
    for meets in (true, false)
        n_eval, restored = Ref(0), Ref(false)
        evaluate!(x) = (n_eval[] += 1; ([1.0], meets && n_eval[] == 2, true))
        iter, outcome = fixed_point_solve!(
            evaluate!, NewtonStepper(lu(fill(1.0, 1, 1)), [1.0]), [0.0];
            max_iter = 10, keep! = () -> nothing, restore! = () -> (restored[] = true),
        )
        @test iter == 2
        @test outcome === (meets ? :converged : :stopped)
        @test restored[] == !meets
    end
end
