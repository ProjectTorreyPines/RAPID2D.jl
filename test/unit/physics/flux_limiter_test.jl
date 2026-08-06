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
        calculate_para_grad_of_scalar_F, calculate_grad_of_scalar_F, flux_limited_diffusivity

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
            # Elongated along R̂, so ∇n has both components and b̂·∇n vanishes on a
            # curve that cuts through live gradient rather than along a flat.
            #
            # Written IN-WALL ONLY, so `initialize!`'s uniform exterior survives. That
            # exterior is the subject of the "flat region is untouched" testitem: an
            # analytic blob evaluated over the whole grid is never bit-exactly flat
            # anywhere, so writing it everywhere would leave that test nothing to assert on.
            R = RP.G.R2D
            Z = RP.G.Z2D
            blob = @. 1.0e18 * exp(-((R - 1.5)^2 / (2 * 0.2^2) + Z^2 / (2 * 0.05^2))) + 1.0e6
            vec(RP.plasma.ne)[RP.G.nodes.in_wall_nids] .= vec(blob)[RP.G.nodes.in_wall_nids]
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

@testitem "Limiter wiring: a uniform profile is untouched" setup = [LimiterRun] begin
    # THE REGRESSION TEST FOR THE OLD GUARD, at the scale it actually operated on.
    # `Lne_para[!isfinite] = 0` mapped ∇∥n = 0 to D = 0, so on a uniform field it
    # zeroed the diffusivity across the WHOLE domain. The new form returns D unchanged.
    #
    # A uniform `ne` is what makes this assertion exact: every node is bit-exactly
    # flat, so the limiter is a total no-op, and `extrapolate_field_to_boundary_nodes!`
    # and the out-of-wall damping then apply identically to both sides and cannot
    # manufacture a difference. Asserting on a flat SUBSET of a blob fixture cannot
    # work — post-stage-3 boundary extrapolation copies in-wall values onto adjacent
    # exterior nodes, so divergence leaks into any subset the field itself defines.
    #
    # This is a real behaviour change and it is the one to watch: electron density is
    # diffused through the ordinary operator and only zeroed outside the wall
    # afterwards (physics.jl:742), so a lagged D_Fick out there lets the implicit
    # matrix move newly arrived material through several exterior cells within one
    # solve before it is deleted and booked as wall loss.
    on = limiter_case(; ne = 1.0e15, limit = true)
    off = limiter_case(; ne = 1.0e15, limit = false)
    @test on.transport.Dpara == off.transport.Dpara
    # not vacuous: the field being compared is substantial, not zero on both sides
    @test maximum(on.transport.Dpara) > 1.0e3
end

@testitem "Limiter wiring: D never depends on the sign of the flow" setup = [LimiterRun] begin
    # `calculate_para_grad_of_scalar_F` defaults to upwind = flags.upwind = true, which
    # picks its one-sided stencil from sign(ueR)/sign(ueZ). A diffusive closure has no
    # business reading the flow direction, and the operator it is bounding uses centred
    # face gradients anyway. Passing upwind = false at that one call site is what this
    # asserts.
    #
    # The fixture has to be OBLIQUE and the seed has to go through `ue_para`:
    # `update_transport_quantities!` recomputes `ueR .= ue_para * bR` before returning,
    # so seeding `ueR` directly is erased, and a purely toroidal field converts any
    # `ue_para` straight back to `ueR = 0`. The second call is what lets the limiter
    # see the seeded flow at all — on the first, it is still reading the stale zeros
    # from before the conversion, which is precisely the lag the comment is about.
    function with_flow(s)
        RP = limiter_case(; oblique = true)
        RP.plasma.ue_para .= s * 1.0e6
        update_transport_quantities!(RP)   # ue_para -> ueR/ueZ; limiter saw stale zeros
        update_transport_quantities!(RP)   # limiter now sees the seeded flow
        return RP
    end
    fwd, rev = with_flow(+1), with_flow(-1)
    @test fwd.transport.Dpara ≈ rev.transport.Dpara rtol = 1.0e-14

    # and prove the fixture can tell the difference — under the upwind stencil the two
    # gradients genuinely differ, so the assertion above is doing real work rather than
    # comparing two copies of the same number
    gf = calculate_para_grad_of_scalar_F(fwd, fwd.plasma.ne; upwind = true)
    gr = calculate_para_grad_of_scalar_F(rev, rev.plasma.ne; upwind = true)
    @test gf != gr
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

# ── behavioural, 1-D ────────────────────────────────────────────────────────
#
# The 1-D problem is the specification (internal/docs/figs/flux_limiter_1d.jl, which
# cannot be included from here because internal/ is gitignored). Backward Euler with D
# lagged one step and arithmetically averaged to faces — the 1-D reduction of
# `update_∇𝐃∇_operator!`.
#
# These are GATES, NOT SELECTORS. Every member of Larsen's family is exact in both
# limits, so n = 1, n = 2 and n = ∞ pass all of them identically (measured: the same
# front ratio, the same 0.3750 mm penetration depth). Nothing here discriminates
# between exponents — only the small-R kinetic comparison in the design doc does. What
# these catch is the limiter being absent, mis-scaled, or applied to the wrong field.

@testsnippet Limiter1D begin
    using LinearAlgebra: Tridiagonal
    using RAPID2D: flux_limited_diffusivity

    const V̄1 = 1.496e6      # ⟨|v|⟩ for 5 eV electrons   [m/s]
    const DF1 = 2.3e4       # T/(mν) at the default fill [m²/s]
    const Λ1 = 2 * DF1 / V̄1  # D = ½v̄λ ⟹ λ = 3.07 cm      [m]

    "∂ₓn by centred differences, one-sided at the ends."
    function ddx(n, dx)
        g = similar(n)
        g[1] = (n[2] - n[1]) / dx
        g[end] = (n[end] - n[end - 1]) / dx
        @views @. g[2:(end - 1)] = (n[3:end] - n[1:(end - 2)]) / (2dx)
        return g
    end

    "One implicit step of ∂ₜn = ∂ₓ(D ∂ₓn) − νn, D lagged and averaged to faces."
    function step1d!(n, dx, dt; limited::Bool, ν = 0.0, bc = :reflect, n₀ = 0.0)
        N = length(n)
        g = ddx(n, dx)
        Dn = limited ?
            [flux_limited_diffusivity(DF1, g[i], n[i], V̄1, 0.25) for i in 1:N] :
            fill(DF1, N)
        Df = [0.5 * (Dn[i] + Dn[i + 1]) for i in 1:(N - 1)]
        lo, di, up = zeros(N - 1), zeros(N), zeros(N - 1)
        for i in 1:N
            fl = i > 1 ? Df[i - 1] / dx^2 : 0.0
            fr = i < N ? Df[i] / dx^2 : 0.0
            di[i] = 1 + dt * (fl + fr + ν)
            i > 1 && (lo[i - 1] = -dt * fl)
            i < N && (up[i] = -dt * fr)
        end
        A = Tridiagonal(lo, di, up)
        rhs = copy(n)
        if bc === :dirichlet
            # Row 1 only. Zeroing A.dl[1] instead would cut the reservoir off from the
            # cell it feeds — a silent no-flux wall.
            A.d[1] = 1.0
            A.du[1] = 0.0
            rhs[1] = n₀
        end
        n .= A \ rhs
        return n
    end
end

@testitem "Limiter 1-D: unlimited diffusion outruns its own mean speed" setup = [Limiter1D] begin
    # A blob spreads as √(2Dt) while its particles travel v̄t on average, and the first
    # beats the second for all t < 2D/v̄² — until the front has moved one mean free
    # path. NOT a causality violation: a Maxwellian has unbounded velocity support, so
    # v̄t is a mean-speed scale rather than a light cone. The measurement is the
    # diagnostic; the word is not.
    N = 1601
    x = range(-0.5, 0.5; length = N)
    dx, dt = step(x), 2.0e-11
    blob = @. exp(-x^2 / (2 * 4.0e-3^2))
    front(n) = (i = findlast(≥(1.0e-3 * maximum(n)), n); i === nothing ? 0.0 : abs(x[i]))
    x0 = front(blob)

    ratio(limited) = begin
        n, worst = copy(blob), 0.0
        for k in 1:1200
            step1d!(n, dx, dt; limited)
            k % 60 == 0 && (worst = max(worst, front(n) / (x0 + V̄1 * k * dt)))
        end
        worst
    end
    @test ratio(false) > 2.5          # measured 2.762
    @test ratio(true) < 1.05          # measured 0.9276
end

@testitem "Limiter 1-D: the shielding layer stops at the flux-limited fixed point" setup = [Limiter1D] begin
    # Reservoir at x = 0, volumetric sink ν everywhere. Diffusion gives δ = √(D/ν); a
    # particle absorbed at rate ν covers at most ~v̄/ν before it dies. The limited
    # answer lands on ¼v̄/ν, and that is a closed-form check rather than a fit: with an
    # exponential profile Lₙ = δ exactly, so D → ¼v̄δ and δ² = D/ν closes on itself.
    N = 4001
    x = range(0, 0.02; length = N)
    dx, ν = step(x), 1.0e9
    depth(limited) = begin
        n = zeros(N)
        for _ in 1:20000
            step1d!(n, dx, 1.0e-12; limited, ν, bc = :dirichlet, n₀ = 1.0)
        end
        x[findfirst(<(n[1] / ℯ), n)]
    end
    fixed_point = 0.25 * V̄1 / ν            # 0.374 mm
    @test depth(true) ≈ fixed_point rtol = 0.01     # measured 0.375 mm
    @test depth(false) > 10 * fixed_point           # measured 4.805 mm, 12.8×
end

@testitem "Limiter 1-D: the diffusive limit reproduces the unlimited solution" setup = [Limiter1D] begin
    # λ/Lₙ → 0 must return Fick untouched. Widening the blob is what does it — raising
    # its AMPLITUDE does not, because D_max = ¼v̄·n/|∇n| is homogeneous of degree zero
    # in n, so a taller blob has exactly the same cap.
    N = 1601
    x = range(-0.5, 0.5; length = N)
    dx, dt = step(x), 2.0e-11
    gap(σ) = begin
        nl = @. exp(-x^2 / (2σ^2))
        nu = copy(nl)
        for _ in 1:400
            step1d!(nl, dx, dt; limited = true)
            step1d!(nu, dx, dt; limited = false)
        end
        maximum(abs.(nl .- nu)) / maximum(nu)
    end
    @test gap(0.2) < 2.0e-3      # σ₀/λ = 6.5,  measured 8.5e-4
    @test gap(0.1) < 5.0e-3      # σ₀/λ = 3.25, measured 2.7e-3
    @test gap(0.004) > 0.5        # σ₀/λ = 0.13, the cap is doing real work
end

@testitem "Limiter 1-D: inventory is conserved and density stays non-negative" setup = [Limiter1D] begin
    N = 1601
    x = range(-0.5, 0.5; length = N)
    dx, dt = step(x), 2.0e-11
    for limited in (false, true)
        n = @. exp(-x^2 / (2 * 4.0e-3^2))
        Σ0 = sum(n)
        for _ in 1:400
            step1d!(n, dx, dt; limited)
        end
        # reflective ends make the stencil a divergence, so constants lie in its kernel
        @test sum(n) ≈ Σ0 rtol = 1.0e-13
        @test minimum(n) ≥ 0
    end
end

# ── behavioural, 2-D — and it must be 2-D ───────────────────────────────────
#
# A scalar test passes while the inverted guard is live: in 1-D the guard is a
# provable no-op, because D is zeroed exactly where the gradient is zero and D only
# ever enters multiplied by that gradient. That argument does not survive the
# anisotropic tensor, where the limiter reads the PARALLEL projection b̂·∇n — which can
# vanish on a one-cell-wide surface while the perpendicular gradient there is alive.
#
# The field must be OBLIQUE. With B along Ẑ, b_R = 0 and hence
# D_RZ = (D∥ − D⊥)·b_R·b_Z ≡ 0, so an axis-aligned test bypasses exactly the cross
# terms that make the 9-point stencil non-monotone.

@testitem "Limiter 2-D: D is untouched where the field runs along a density contour" setup = [LimiterRun] begin
    # The regression test for the guard, in the geometry where 1-D cannot see it.
    #
    # In 1-D the inverted guard is a provable no-op: D is zeroed exactly where the
    # gradient is zero, and D only ever enters multiplied by that gradient. That
    # argument dies in the anisotropic tensor, where the limiter reads the PARALLEL
    # projection b̂·∇n — which can vanish while the full gradient is alive.
    #
    # Constructed so that vanishing is EXACT rather than nearly so: the density varies
    # along Ẑ only, so ∂n/∂R is a difference of equal values and is exactly 0.0 at
    # every node; pointing b̂ along R̂ then makes b̂·∇n = 1·0 + 0·∂n/∂Z identically zero
    # while |∇n| stays alive everywhere. Hunting instead for isolated nodes that happen
    # to sit on a tangent CURVE cannot be made exact — the curve generically passes
    # between grid nodes, so it would mean fitting the fixture's widths to the mesh and
    # re-fitting them whenever NR, NZ, the wall box or b̂ moved.
    #
    # D_RZ = (D∥ − D⊥)·b_R·b_Z is zero here because b̂ lies on an axis. That is fine:
    # the cross term is the subject of the next testitem, which uses the oblique
    # fixture. Exactness and obliqueness pull in opposite directions, so they are
    # asserted separately rather than compromised into one weaker test.
    function tangent_case(limit)
        RP = limiter_case(; limit)
        @. RP.plasma.ne = 1.0e18 * exp(-RP.G.Z2D^2 / (2 * 0.1^2)) + 1.0e6
        RP.plasma.ni .= RP.plasma.ne
        RP.fields.bR .= 1.0
        RP.fields.bZ .= 0.0
        RP.fields.bϕ .= 0.0
        update_transport_quantities!(RP)
        return RP
    end
    on, off = tangent_case(true), tangent_case(false)

    g = calculate_para_grad_of_scalar_F(on, on.plasma.ne; upwind = false)
    gR, gZ = calculate_grad_of_scalar_F(on, on.plasma.ne; upwind = false)
    full = @. sqrt(gR^2 + gZ^2)

    # the construction holds: parallel projection dead everywhere, full gradient alive
    @test all(iszero, g)
    @test all(>(0), full)
    # so the limiter has nothing to cap, and must return D untouched — the old guard
    # forced 0 at every one of these nodes instead
    @test on.transport.Dpara == off.transport.Dpara
    # not vacuous: the field being compared is substantial, not zero on both sides
    @test maximum(on.transport.Dpara) > 1.0e3
end

@testitem "Limiter 2-D: the oblique field genuinely exercises the cross term" setup = [LimiterRun] begin
    # Guards the test above against silently degenerating: if D_RZ were zero the
    # anisotropic stencil would collapse to 5 points and the geometry that motivates a
    # 2-D test would be gone.
    oblique = limiter_case(; oblique = true)
    inw = oblique.G.nodes.in_wall_nids
    @test maximum(abs, vec(oblique.transport.DRZ)[inw]) > 0

    # and the axis-aligned case that the design originally proposed does not
    axis = limiter_case(; oblique = false)
    axis.fields.bR .= 0.0
    axis.fields.bZ .= 1.0
    axis.fields.bϕ .= 0.0
    update_transport_quantities!(axis)
    @test all(==(0), axis.transport.DRZ)
end
