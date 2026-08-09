@testitem "exprb_B: the identity B(-z)/B(z) = exp(z)" begin
    using RAPID2D: exprb_B

    # B(z) = z/(e^z − 1) is the coefficient that makes the local update exact for
    # the frozen-linear problem dy/dt = λy + S. The whole scheme is
    #     y^{n+1} = yⁿ + Δt·f(yⁿ)/B(λΔt),
    # i.e. forward Euler with B as a divisor on the increment, so this one
    # identity IS the scheme's correctness statement. Aggregated to one
    # assertion: 4001 separate @tests bury a real failure in scroll.
    function worst_identity_error()
        worst, worst_z = 0.0, 0.0
        for z in range(-200.0, 200.0, 4001)
            iszero(z) && continue
            err = Float64(abs(big(exprb_B(-z) / exprb_B(z)) - exp(big(z))) / exp(big(z)))
            err > worst && ((worst, worst_z) = (err, z))
        end
        return worst, worst_z
    end
    worst, worst_z = worst_identity_error()
    @test worst < 1.0e-14        # measured 3.6e-16, at z ≈ −16.6
    worst < 1.0e-14 || @info "B(-z)/B(z) worst case" worst worst_z
end

@testitem "exprb_B: z/expm1(z) is accurate at zero — no series branch needed" begin
    using RAPID2D: exprb_B

    # The design note prescribes series expansions near z = 0 (B ≈ 1 − z/2 + z²/12).
    # They are not needed: expm1 exists precisely to keep e^z − 1 accurate when it
    # cancels, so the quotient inherits that. A truncated series would be worse.
    Bbig(z) = (zb = big(z); iszero(zb) ? big(1.0) : zb / expm1(zb))
    function worst_bigfloat_error()
        worst, worst_z = 0.0, 0.0
        for z in vcat(-10 .^ range(-18, 2.3, 400), 10 .^ range(-18, 2.3, 400))
            err = Float64(abs(big(exprb_B(z)) - Bbig(z)) / abs(Bbig(z)))
            err > worst && ((worst, worst_z) = (err, z))
        end
        return worst, worst_z
    end
    worst, worst_z = worst_bigfloat_error()
    @test worst < 1.0e-15        # measured 1.9e-16
    worst < 1.0e-15 || @info "B(z) vs BigFloat worst case" worst worst_z

    # B(0) = 1 exactly is what makes ForwardEuler a bit-for-bit fallback rather
    # than an approximation of one: a term handed z = 0 contributes nothing.
    @test exprb_B(0.0) === 1.0
    @test exprb_B(-0.0) === 1.0
    @test exprb_B(1.0e-300) == 1.0
end

@testitem "exprb_B: strictly positive, so the diagonal can never change sign" begin
    using RAPID2D: exprb_B

    # This matters more than conditioning. A θ-scheme's 1 − θz goes negative on a
    # growth cell once θz > 1 and the matrix stops being an M-matrix; no amount of
    # rescaling repairs that. B(z) > 0 always, at every Δt.
    zs = vcat(range(-300.0, 300.0, 2001), [-1.0e-30, 1.0e-30, 0.0])
    @test all(isfinite, exprb_B.(zs))
    @test all(>(0), exprb_B.(zs))

    # Monotone decreasing: more stiffness ⇒ more damping of the FE increment.
    @test all(<(0), diff(exprb_B.(range(-50.0, 30.0, 500))))

    # The two limits the design note reads physically.
    @test exprb_B(-50.0) ≈ 50.0 rtol = 1.0e-12   # z → −∞: B → |z|, cell fully relaxed
    @test exprb_B(30.0) ≈ 30 / expm1(30) rtol = 1.0e-14   # = 2.81e-12
    @test exprb_B(30.0) < 1.0e-11                # z → +∞: B → 0, memory term dominates
end

@testitem "exprb_cap_exponent: the diagonal must not be annihilated" begin
    using RAPID2D: exprb_B, exprb_cap_exponent, EXPRB_MAX_EXPONENT

    # expm1 overflows at z ≈ 709 (Float64) / 88 (Float32), and B then returns
    # exactly 0.0 — the diagonal vanishes and the row is left to transport alone.
    @test exprb_B(710.0) == 0.0
    @test exprb_B(89.0f0) == 0.0f0

    # The cap sits far below that, where conditioning starts to cost digits
    # rather than where arithmetic fails: the spread goes like
    # |z_dec|·e^(z_gro)/z_gro, so z_gro ≈ 16 already puts κ near 5e7.
    @test EXPRB_MAX_EXPONENT == 30
    @test exprb_cap_exponent(1.0e4) == 30.0
    @test exprb_cap_exponent(5.0) == 5.0
    @test exprb_cap_exponent(-1.0e6) == -1.0e6      # decay is never capped: B → |z| is benign
    @test exprb_cap_exponent(1.0f4) === 30.0f0      # type-preserving

    # A capped z still yields a usable (positive, finite) diagonal — at either
    # width, since RAPID2D is FT-generic and Float32 overflows 40× sooner.
    @test 0 < exprb_B(exprb_cap_exponent(1.0e6)) < 1.0e-11
    b32 = exprb_B.(exprb_cap_exponent.(Float32.(range(-100.0, 100.0, 501))))
    @test eltype(b32) === Float32 && all(isfinite, b32) && all(>(0), b32)
    @test exprb_B(0.0f0) === 1.0f0
end

@testitem "exprb_B: forming θ is what this avoids" begin
    using RAPID2D: exprb_B

    # The design note warns that building the scheme from θ(z) = 1/z − 1/(eᶻ−1)
    # destroys it: the diagonal 1 − θz is a subtraction that cancels as θz → 1.
    # Recorded as a test so nobody reintroduces the θ route as a "clearer"
    # refactor — B(z) is not an optimization of it, it is the only stable form.
    θ_route(z) = 1 - (1 / z - 1 / (exp(z) - 1)) * z
    @test θ_route(36.0) / exprb_B(36.0) - 1 > 0.005    # ~1 % wrong already
    @test !isfinite(θ_route(800.0)) || θ_route(800.0) <= 0 # gone entirely
    @test exprb_B(800.0) >= 0                          # B just underflows, cleanly
end
