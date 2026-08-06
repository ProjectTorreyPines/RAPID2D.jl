# The free-streaming ceiling, on its own — no grid, no RAPID object, no solver.
#
# Every assertion here is closed-form checkable, so these pin physics rather than a
# previous implementation. The closure is
#
#     1/D_limited = √( (1/D)² + (|∇n| / (α·v̄·n))² )        Larsen n = 2
#
# and the whole point of writing it in inverse space is that Lₙ = n/|∇n| is never
# formed: the only division left is by n, and its divergent case (n = 0 with a live
# gradient ⟹ D = 0) is the physically wanted answer rather than a failure.
#
# Why n = 2 and not the n = 1 of the other channels: the exponent IS the order of the
# leading correction, D_n/D_Fick = 1 − x^n/n, and the exact BGK answer for this code's
# fixed-ν model is second order (1 − 1.910R²). See notes/design/flux-limiter.md §1.

@testsnippet FluxCeiling begin
    using RAPID2D: flux_limited_diffusivity, maxwellian_mean_speed

    # The RAPID2D breakdown regime: 5 eV electrons, the default gas fill.
    const V̄ = 1.496e6      # ⟨|v|⟩ = √(8T/πm)          [m/s]
    const D_FICK = 2.3e4   # T/(mν)                    [m²/s]
    const α = 0.25         # the Hertz-Knudsen ¼

    "The ceiling as a diffusivity, for cross-checking: α·v̄·Lₙ."
    D_max(∇n, n) = α * V̄ * n / abs(∇n)
end

@testitem "Flux ceiling: a flat profile is returned untouched, bit for bit" setup = [FluxCeiling] begin
    # THIS IS THE REGRESSION TEST FOR THE OLD GUARD. `Lne_para[!isfinite] = 0` mapped
    # ∇∥n = 0 to D = 0 — maximum limit where the correct reading is NO limit. The new
    # form short-circuits before any arithmetic, so the return is `===` identical, not
    # merely approximate: a round trip through inv(hypot(inv(D), 0)) would lose an ulp.
    for D in (2.3e4, 1.0e-6, 1.0e12, floatmax())
        @test flux_limited_diffusivity(D, 0.0, 1.0e18, V̄, α) === D
    end
    # and it must not depend on the density there, including at n = 0
    @test flux_limited_diffusivity(D_FICK, 0.0, 0.0, V̄, α) === D_FICK
end

@testitem "Flux ceiling: an empty cell supplies no flux" setup = [FluxCeiling] begin
    # n = 0 with a live gradient is the one case where D = 0 is right: there are no
    # particles to carry a flux. It must be exactly zero and finite — never NaN, which
    # is what n/∇n would have produced.
    @test flux_limited_diffusivity(D_FICK, 1.0e20, 0.0, V̄, α) === 0.0
    # a transiently negative density (negative_n_correction has not run yet) takes the
    # same branch rather than silently behaving like |n|
    @test flux_limited_diffusivity(D_FICK, 1.0e20, -1.0e6, V̄, α) === 0.0
    # T = 0 ⟹ v̄ = 0 ⟹ nothing crosses any surface
    @test flux_limited_diffusivity(D_FICK, 1.0e20, 1.0e18, 0.0, α) === 0.0
    # and a zero diffusivity stays zero
    @test flux_limited_diffusivity(0.0, 1.0e20, 1.0e18, V̄, α) === 0.0
end

@testitem "Flux ceiling: both limits are exact" setup = [FluxCeiling] begin
    # D_Fick → ∞: the ceiling alone survives, D → α·v̄·Lₙ. floatmax() and Inf must both
    # land there — floatmax() because 1/floatmax() is subnormal and a naive
    # D/√(1+(D/D_max)²) would overflow to zero instead.
    ∇n, n = 1.0e20, 1.0e18
    for D in (1.0e30, floatmax(), Inf)
        @test flux_limited_diffusivity(D, ∇n, n, V̄, α) ≈ D_max(∇n, n) rtol = 1.0e-12
    end
    # Lₙ → ∞: the collisional value alone survives
    @test flux_limited_diffusivity(D_FICK, 1.0e-30, n, V̄, α) ≈ D_FICK rtol = 1.0e-12
end

@testitem "Flux ceiling: the composition is Larsen n = 2" setup = [FluxCeiling] begin
    # 1/D = √((1/D_Fick)² + (1/D_max)²) — NOT the reciprocal sum the other channels use.
    # A gradient is not a competing collision event, so this is a ceiling on the total
    # rather than a term in the Matthiessen sum; the arithmetic differs for that reason.
    for (∇n, n) in ((1.0e20, 1.0e18), (1.0e17, 1.0e18), (5.0e21, 3.0e17))
        got = flux_limited_diffusivity(D_FICK, ∇n, n, V̄, α)
        @test inv(got) ≈ hypot(inv(D_FICK), inv(D_max(∇n, n))) rtol = 1.0e-12
        # and it is strictly below both parents, as every member of the family is
        @test got < D_FICK
        @test got < D_max(∇n, n)
    end
end

@testitem "Flux ceiling: monotone in both arguments" setup = [FluxCeiling] begin
    n = 1.0e18
    steeper = [flux_limited_diffusivity(D_FICK, g, n, V̄, α) for g in (1.0e18, 1.0e19, 1.0e20, 1.0e21)]
    @test issorted(steeper; rev = true)          # steeper gradient ⟹ tighter cap
    denser = [flux_limited_diffusivity(D_FICK, 1.0e20, ni, V̄, α) for ni in (1.0e16, 1.0e17, 1.0e18, 1.0e19)]
    @test issorted(denser)                        # more supply ⟹ looser cap
end

@testitem "Flux ceiling: D is smooth where the gradient vanishes" setup = [FluxCeiling] begin
    # n = 2 depends on (∇n)², so ∂D/∂∇n agrees from both sides at ∇n = 0. n = 1 depends
    # on |∇n| and does not — and ∇∥n ≈ 0 is where most of the grid sits most of the
    # time, so this is the common case, not an edge case. This test is the reason the
    # exponent is 2 rather than 1 as an earlier revision of the design proposed.
    n = 1.0e18
    for h in (1.0e14, 1.0e15, 1.0e16)
        @test flux_limited_diffusivity(D_FICK, h, n, V̄, α) ===
            flux_limited_diffusivity(D_FICK, -h, n, V̄, α)
    end
    # the one-sided slopes converge to each other as h → 0 (they do not for n = 1,
    # where they converge to ∓α⁻¹ and differ by a factor of −1)
    slope(h) = (flux_limited_diffusivity(D_FICK, h, n, V̄, α) - D_FICK) / h
    @test abs(slope(1.0e15)) < abs(slope(1.0e18))     # vanishing, i.e. second order
end

@testitem "Flux ceiling: the moment is the mean speed, not the most probable one" setup = [FluxCeiling] begin
    using RAPID2D: maxwellian_most_probable_speed
    # The old code used vp = √(2T/m) where Hertz-Knudsen wants v̄ = √(8T/πm). The
    # ceiling is a one-way flux across a surface and only v̄ has that meaning; the
    # ratio is √(4/π) = 1.1284, so the old cap ran 12.8 % tighter than physical.
    Te, me = 5.0, 9.1093837015e-31
    vm = maxwellian_mean_speed(Te, me)
    vp = maxwellian_most_probable_speed(Te, me)
    @test vm / vp ≈ sqrt(4 / π) rtol = 1.0e-12
    ∇n, n = 1.0e21, 1.0e18            # deep in the capped regime so the ratio shows
    @test flux_limited_diffusivity(1.0e30, ∇n, n, vm, α) /
        flux_limited_diffusivity(1.0e30, ∇n, n, vp, α) ≈ sqrt(4 / π) rtol = 1.0e-10
end
