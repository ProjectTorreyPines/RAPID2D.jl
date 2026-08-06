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

# ── the wiring ──────────────────────────────────────────────────────────────
#
# Three stages, and the order is the physics:
#
#   1  1/D_ch = Σ_p 1/D_p        competing termination events    (PR #11, unchanged)
#   2  D∥ = D_ch + Dpara0        independent arrival paths       (unchanged)
#   3  cap(D∥, ¼·v̄·Lₙ)           a causality bound on the TOTAL  (new, LAST)
#
# Stage 3 has to come last because the ceiling is mechanism-agnostic: it bounds the
# total parallel flux whatever carries it. The old code capped `Dpara_e_eff` and then
# ADDED `Dpara0`, so the floor escaped the bound entirely.

@testsnippet LimiterRun begin
    using RAPID2D: update_transport_quantities!, maxwellian_mean_speed,
        calculate_para_grad_of_scalar_F, flux_limited_diffusivity

    "A walled discharge-shaped case with an elongated blob and a settable field."
    function limiter_case(;
            NR = 41, NZ = 41, Te = 5.0, Dpara0 = 0.0,
            limit = true, oblique = false, ne = nothing
        )
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            Dpara0 = Dpara0,
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.flags.limit_flux = (state = limit, factor = 0.25)
        RP.plasma.Te_eV .= Te
        RP.plasma.Ti_eV .= 1.0
        if ne === nothing
            # elongated along R̂, so ∇n has both components and b̂·∇n vanishes on a
            # curve that cuts through live gradient rather than along a flat
            R = RP.G.R2D
            Z = RP.G.Z2D
            @. RP.plasma.ne = 1.0e18 * exp(-((R - 1.5)^2 / (2 * 0.2^2) + Z^2 / (2 * 0.05^2))) + 1.0e6
        else
            RP.plasma.ne .= ne
        end
        RP.plasma.ni .= RP.plasma.ne
        if oblique
            # 3-4-5 in the poloidal plane with a toroidal part, so b_R·b_Z ≠ 0 and
            # therefore D_RZ ≠ 0. An axis-aligned field gives D_RZ ≡ 0 and would
            # bypass the cross terms entirely.
            bR, bZ, bϕ = 0.36, 0.48, 0.8       # norm 1
            RP.fields.bR .= bR
            RP.fields.bZ .= bZ
            RP.fields.bϕ .= bϕ
        end
        update_transport_quantities!(RP)
        return RP
    end
end

@testitem "Limiter wiring: the ceiling binds Dpara0, not just the collisional part" setup = [LimiterRun] begin
    # The old code clamped `Dpara_e_eff` and added `Dpara0` afterwards, so a floor
    # larger than the ceiling transported faster than free streaming. Latent at the
    # config default of 0.0 — but the `Transport` struct's OWN default is 1.0
    # (types.jl:375), so this is one constructor away from being live.
    #
    # A floor that already exceeds the ceiling everywhere makes the leak unmissable.
    RP = limiter_case(; Dpara0 = 1.0e12, limit = true)
    tp, pla = RP.transport, RP.plasma
    vm = maxwellian_mean_speed.(pla.Te_eV, RP.config.constants.me)
    g = abs.(calculate_para_grad_of_scalar_F(RP, pla.ne; upwind = false))
    inw = RP.G.nodes.in_wall_nids

    # wherever the gradient is live, D must sit under the supply ceiling
    live = [k for k in inw if vec(g)[k] > 0 && vec(pla.ne)[k] > 0]
    @test !isempty(live)
    for k in live
        @test vec(tp.Dpara)[k] * vec(g)[k] <= 0.25 * vec(pla.ne)[k] * vec(vm)[k] * (1 + 1.0e-9)
    end
    # and the floor is genuinely being cut down, i.e. the test is not vacuous
    @test minimum(vec(tp.Dpara)[live]) < 1.0e12
end

@testitem "Limiter wiring: a flat region is untouched, so the out-of-wall D changes" setup = [LimiterRun] begin
    # `initialize!` leaves the out-of-wall state UNIFORM, so ∇∥n = 0 there. The old
    # guard mapped that to D = 0 across the whole exterior; the new form returns D
    # unchanged. Asserting "identical with the limiter on and off" pins the guard
    # removal exactly, and is a stronger statement than measuring the difference.
    #
    # This is a real behaviour change and it is the one to watch: electron density is
    # diffused through the ordinary operator and only zeroed outside the wall
    # afterwards (physics.jl:742), so a lagged D_Fick out there lets the implicit
    # matrix move newly arrived material through several exterior cells within one
    # solve before it is deleted and booked as wall loss.
    on = limiter_case(; limit = true)
    off = limiter_case(; limit = false)
    out = setdiff(1:length(on.plasma.ne), on.G.nodes.in_wall_nids)
    @test !isempty(out)
    @test vec(on.transport.Dpara)[out] == vec(off.transport.Dpara)[out]
    # and it is not trivially zero on both sides
    @test any(>(0), vec(on.transport.Dpara)[out])
end

@testitem "Limiter wiring: D never depends on the sign of the flow" setup = [LimiterRun] begin
    # `calculate_para_grad_of_scalar_F` defaults to upwind = flags.upwind = true, which
    # picks its one-sided stencil from sign(ueR)/sign(ueZ) — and from values refreshed
    # LATER in the same function than the limiter reads them. A diffusive closure has
    # no business reading the flow direction, and the operator it is bounding uses
    # centred face gradients anyway. Passing upwind = false at that one call site is
    # what this asserts.
    fwd = limiter_case()
    rev = limiter_case()
    rev.plasma.ue_para .= .-rev.plasma.ue_para
    rev.plasma.ueR .= .-rev.plasma.ueR
    rev.plasma.ueZ .= .-rev.plasma.ueZ
    update_transport_quantities!(rev)
    @test fwd.transport.Dpara ≈ rev.transport.Dpara rtol = 1.0e-14
end

@testitem "Limiter wiring: Lₙ reads the raw density the solver transports" setup = [LimiterRun] begin
    # The old form was Lₙ = n_raw / |∇∥(n_smoothed)| — two different fields. On the
    # low-density flank of a one-cell front that made the cap ~5× tighter than the
    # consistent form. Smoothing both would fix the inconsistency but would bound
    # D·|∇n_SM| while the solve transports raw `pla.ne`, so the smoothing is dropped
    # instead; consistent smoothing moves 1/Lₙ by only ~1 % anyway.
    #
    # Asserted by reproducing the whole stage-3 map from raw fields alone.
    RP = limiter_case()
    tp, pla = RP.transport, RP.plasma
    vm = maxwellian_mean_speed.(pla.Te_eV, RP.config.constants.me)
    g = calculate_para_grad_of_scalar_F(RP, pla.ne; upwind = false)
    expected = @. flux_limited_diffusivity(tp.Dpara0 + tp.Dpara_e_eff, g, pla.ne, vm, 0.25)
    # compare in-wall only: outside, the damping function and the boundary
    # extrapolation both rewrite Dpara afterwards
    inw = RP.G.nodes.in_wall_nids
    @test vec(tp.Dpara)[inw] ≈ vec(expected)[inw] rtol = 1.0e-12
end
