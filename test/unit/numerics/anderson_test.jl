# Anderson mixing for a fixed point x = g(x), apart from any physics: given an iterate and its
# residual f = g(x) − x it returns the next iterate, keeps the best one, and restarts from it when
# a residual grows or is not finite.

@testitem "Anderson mixing solves an affine fixed point that relaxed iteration diverges on" begin
    using RAPID2D: AndersonMixer, anderson_step!
    using LinearAlgebra
    # T with eigenvalues −5, −2 and 0.95 among seventeen in [0, 0.5], in a fixed orthogonal basis:
    # the relaxed iteration with β = 1/2 multiplies the first mode by 1 − (1 − μ)/2 = −2 each step,
    # and the 0.95 mode by 0.975.
    n = 20
    λ = vcat([-5.0, -2.0, 0.95], 0.5 .* (1:(n - 3)) ./ (n - 3))
    Q = Matrix(qr([sin(i * j + i) for i in 1:n, j in 1:n]).Q)
    T = Q * Diagonal(λ) * Q'
    c = cos.(1:n)
    x_star = (I - T) \ c
    g(x) = T * x + c

    function relaxed(x, nsteps)
        for _ in 1:nsteps
            x = x .+ 0.5 .* (g(x) .- x)
        end
        return x
    end
    @test norm(relaxed(zeros(n), 30) .- x_star) > 1.0e3 * norm(x_star)   # relaxed: diverges

    # Anderson: the number of steps to 1e-12 (measured 30 with m = 8; 25 with the whole history,
    # m = n, where it works as GMRES and needs about n steps)
    function anderson(A, x)
        for k in 1:100
            norm(x .- x_star) <= 1.0e-12 * norm(x_star) && return x, k - 1
            x, _ = anderson_step!(A, x, g(x) .- x)
        end
        return x, 100
    end
    for (m, bound) in ((8, 40), (n, n + 10))
        A = AndersonMixer{Float64}(n, m; β = fill(0.5, n))
        x, nsteps = anderson(A, zeros(n))
        @test norm(x .- x_star) <= 1.0e-12 * norm(x_star)
        @test nsteps <= bound
        @test A.nrestart == 0
    end
end

@testitem "Anderson mixing: m = 0 is the relaxed iteration" begin
    using RAPID2D: AndersonMixer, anderson_step!
    β = [0.5, 0.5, 1.0, 1.0]
    A = AndersonMixer{Float64}(4, 0; β)
    x0, f0 = [1.0, 2.0, 3.0, 4.0], [0.1, -0.2, 0.3, -0.4]
    x1, s1 = anderson_step!(A, x0, f0)
    @test s1 === :best
    @test x1 == x0 .+ β .* f0
    x2, s2 = anderson_step!(A, x1, f0 ./ 2)
    @test s2 === :best
    @test x2 == x1 .+ β .* (f0 ./ 2)
end

@testitem "Anderson mixing: a grown residual restarts from the best iterate with half the mixing" begin
    using RAPID2D: AndersonMixer, anderson_step!
    A = AndersonMixer{Float64}(3, 4; β = fill(0.5, 3))
    xb, fb = [1.0, 1.0, 1.0], [1.0e-3, -1.0e-3, 2.0e-3]
    @test anderson_step!(A, xb, fb)[2] === :best
    x2, s = anderson_step!(A, [5.0, 5.0, 5.0], [10.0, 10.0, 10.0])   # 4e3 times the best residual
    @test s === :restart
    @test A.nrestart == 1
    @test x2 == xb .+ 0.25 .* fb
    # the history went with it: the next step is a relaxed one with the halved mixing
    x3, s3 = anderson_step!(A, x2, fb ./ 2)
    @test s3 === :best
    @test x3 == x2 .+ 0.25 .* (fb ./ 2)
end

@testitem "Anderson mixing: a residual that is not finite is never kept" begin
    using RAPID2D: AndersonMixer, anderson_step!, best_iterate
    A = AndersonMixer{Float64}(3, 4; β = fill(0.5, 3))
    xb, fb = [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]
    anderson_step!(A, xb, fb)
    x2, s = anderson_step!(A, [9.0, 9.0, 9.0], [NaN, 1.0, 1.0])
    @test s === :restart
    @test all(isfinite, x2)
    @test best_iterate(A) == xb
    # with no finite iterate yet there is nothing to restart from
    @test anderson_step!(AndersonMixer{Float64}(3, 4), [1.0, 1.0, 1.0], [Inf, 0.0, 0.0])[2] === :failed
end

@testitem "Anderson mixing: dependent differences are dropped" begin
    using RAPID2D: AndersonMixer, anderson_step!
    A = AndersonMixer{Float64}(3, 4; β = fill(0.5, 3))
    x, f = [1.0, 2.0, 3.0], [0.1, 0.2, 0.3]
    for _ in 1:5   # the same pair again and again: every difference is zero
        x_next, _ = anderson_step!(A, x, f)
        @test x_next ≈ x .+ 0.5 .* f
    end
end

@testitem "Anderson mixing: with its restarts used up it stops at the best iterate" begin
    using RAPID2D: AndersonMixer, anderson_step!
    A = AndersonMixer{Float64}(2, 2; β = fill(0.5, 2), max_restarts = 2)
    xb, fb = [1.0, 1.0], [1.0e-6, 1.0e-6]
    anderson_step!(A, xb, fb)
    steps = [anderson_step!(A, [2.0, 2.0], [1.0, 1.0]) for _ in 1:3]
    @test last.(steps) == [:restart, :restart, :exhausted]
    @test first(steps[end]) == xb
end
