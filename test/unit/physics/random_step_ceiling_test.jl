# The geometric random-step ceiling: D = v̄·L/d, and the wall distance that feeds it.
#
# Every geometric bound on a diffusivity in this code has that one form; the only thing
# that differs is the denominator, which is an angular average of the SAME v̄:
#
#     d = 3   ⟨v_z²/|v|⟩ = v̄/3   isotropic — a neutral molecule
#     d = 2   ⟨|v_x|⟩    = v̄/2   along one axis — a magnetized particle on b̂
#
# derived in internal/docs/src/details/speed-and-composition.md.
#
# The failure these tests exist to prevent is `0 * Inf = NaN`. Both a zero speed and an
# infinite length are ordinary states here — a cold species, a closed field line — and
# their combination must land on a defined answer rather than poisoning a solve.

@testitem "Random-step algebra: the denominator is the angular average" begin
    using RAPID2D: geometric_diffusivity, inv_geometric_diffusivity,
        IsotropicStep, AlongAxisStep, step_denominator

    @test step_denominator(IsotropicStep()) == 3
    @test step_denominator(AlongAxisStep()) == 2

    # Against independently written oracles, not against each other.
    v, L = 1500.0, 0.4
    @test geometric_diffusivity(v, L, IsotropicStep()) ≈ 1500.0 * 0.4 / 3
    @test geometric_diffusivity(v, L, AlongAxisStep()) ≈ 1500.0 * 0.4 / 2
    @test inv_geometric_diffusivity(v, L, IsotropicStep()) ≈ 3 / (1500.0 * 0.4)
    @test inv_geometric_diffusivity(v, L, AlongAxisStep()) ≈ 2 / (1500.0 * 0.4)

    # The two forms are reciprocals wherever both are finite and nonzero.
    @test geometric_diffusivity(v, L, IsotropicStep()) *
        inv_geometric_diffusivity(v, L, IsotropicStep()) ≈ 1.0
end

@testitem "Random-step algebra: zero and infinity have a defined precedence" begin
    using RAPID2D: geometric_diffusivity, inv_geometric_diffusivity,
        IsotropicStep, AlongAxisStep

    for FT in (Float32, Float64)
        z, f, i = zero(FT), FT(2), FT(Inf)
        for model in (IsotropicStep(), AlongAxisStep())
            # ZERO WINS OVER INFINITY. A zero speed means nothing moves; a zero length
            # means nothing gets anywhere. Either way the answer is no diffusion — and
            # the inverse form must then say Inf, so that a mechanism which cannot act
            # shuts its limb of a reciprocal sum instead of contributing a NaN.
            for (v, L) in ((z, z), (z, f), (z, i), (f, z), (i, z))
                @test geometric_diffusivity(v, L, model) === zero(FT)
                @test inv_geometric_diffusivity(v, L, model) === FT(Inf)
            end
            # Both positive with either infinite: unbounded, and exactly 0 inverse.
            for (v, L) in ((f, i), (i, f), (i, i))
                @test geometric_diffusivity(v, L, model) === FT(Inf)
                @test inv_geometric_diffusivity(v, L, model) === zero(FT)
            end
            # The type must not widen: a Float32 grid stays Float32.
            @test geometric_diffusivity(f, FT(5), model) isa FT
            @test inv_geometric_diffusivity(f, FT(5), model) isa FT
            # No valid input anywhere in the product may return NaN.
            for v in (z, f, i), L in (z, f, i)
                @test !isnan(geometric_diffusivity(v, L, model))
                @test !isnan(inv_geometric_diffusivity(v, L, model))
            end
        end
    end
end

@testitem "Random-step algebra: a negative or NaN input is a programming error" begin
    using RAPID2D: geometric_diffusivity, inv_geometric_diffusivity,
        wall_step_ceiling, IsotropicStep

    # A negative product passes straight through `min` and reads as a very tight
    # ceiling, so it would silently zero out transport rather than announce itself.
    @test_throws DomainError geometric_diffusivity(-1.0, 2.0, IsotropicStep())
    @test_throws DomainError geometric_diffusivity(2.0, -1.0, IsotropicStep())
    @test_throws DomainError inv_geometric_diffusivity(-1.0, 2.0, IsotropicStep())
    @test_throws DomainError geometric_diffusivity(NaN, 2.0, IsotropicStep())
    @test_throws DomainError geometric_diffusivity(2.0, NaN, IsotropicStep())
    @test_throws DomainError wall_step_ceiling(1.0, -1.0, 2.0)
    @test_throws DomainError wall_step_ceiling(1.0, 2.0, NaN)
end

@testitem "Wall ceiling: forward and backward are disjoint half-populations" begin
    using RAPID2D: wall_step_ceiling, geometric_diffusivity, AlongAxisStep

    v = 1000.0

    # THE ARITHMETIC MEAN, NOT THE HARMONIC ONE. Lf and Lb bound different halves of
    # velocity space — a particle moving forward can only be stopped by the forward
    # wall — so they are not competing termination events. Integrating the one-way flux
    # over each half separately gives (v̄/4)(Lf + Lb).
    @test wall_step_ceiling(v, 3.0, 7.0) ≈ v * (3.0 + 7.0) / 4
    # A harmonic mean would give 2·3·7/10 = 4.2 ⟹ v·2.1 against the correct v·2.5.
    # Pin the gap, or the two compositions are indistinguishable at symmetric lengths.
    @test !isapprox(wall_step_ceiling(v, 3.0, 7.0), v * 4.2 / 2)

    # Symmetric lengths recover the textbook ½v̄L, which is the sanity check that the
    # ¼ is the ½ applied to a mean rather than a third independent coefficient.
    @test wall_step_ceiling(v, 5.0, 5.0) ≈ geometric_diffusivity(v, 5.0, AlongAxisStep())
    # Direction is a labelling choice, so the ceiling cannot depend on the order.
    @test wall_step_ceiling(v, 3.0, 7.0) === wall_step_ceiling(v, 7.0, 3.0)

    # One direction already at the wall still lets the other half stream: exactly half
    # the contribution, not zero and not the full symmetric value.
    @test wall_step_ceiling(v, 0.0, 8.0) ≈ v * 8.0 / 4
    # Both at the wall is the exterior, where the answer is exactly no transport.
    @test wall_step_ceiling(v, 0.0, 0.0) === 0.0
    # A dead species does not move however open the geometry is.
    @test wall_step_ceiling(0.0, Inf, Inf) === 0.0
    @test wall_step_ceiling(0.0, 5.0, 5.0) === 0.0
    # One direction never reaching a wall leaves that half-population unbounded, so
    # there is no ceiling at all — even though the other half is bounded.
    @test wall_step_ceiling(v, 4.0, Inf) === Inf
    @test wall_step_ceiling(v, Inf, Inf) === Inf
end

@testitem "Neutral gas: moving the Knudsen term onto the helper changes no bit" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_channel, h2_self_diffusivity,
        maxwellian_mean_speed, M_H2_GAS, NIST_H2_N_REF, EE

    # The pre-refactor expression, transcribed verbatim from neutral_gas.jl before the
    # change. `≈` would not detect the failure this guards against: if the helper were
    # written as inv(geometric_diffusivity(...)) the round trip through a division and
    # back would cost an ulp, which is invisible to a tolerance and visible to `===`.
    function oracle_scalar(n_gas, T_gas_eV, ν_iz, L_char)
        vm_g = maxwellian_mean_speed(T_gas_eV, M_H2_GAS)
        D_elastic = h2_self_diffusivity(T_gas_eV) * NIST_H2_N_REF / n_gas
        inv_D = 1 / D_elastic +
            M_H2_GAS * ν_iz / (T_gas_eV * EE) +
            (vm_g > 0 ? 3 / (vm_g * L_char) : oftype(vm_g, Inf))
        return 1 / inv_D
    end

    # n_gas = 0 is a burnt-out cell, L_char = Inf disables the wall term, ν_iz = 0 is
    # no ionization: every degenerate limb the production code can actually reach.
    for n_gas in (0.0, 1.0e18, 1.0e22), T in (0.026, 0.5), ν in (0.0, 1.0e4),
            L in (0.1, 1.0, Inf)
        @test neutral_gas_diffusivity(n_gas, T, ν, L) === oracle_scalar(n_gas, T, ν, L)
    end

    # The channel form carries six fields, and `===` on the struct compares identity
    # rather than value — every field has to be compared on its own.
    function oracle_channel(n_gas, T_gas_eV, ν_iz, L_char)
        vm_g = maxwellian_mean_speed.(T_gas_eV, M_H2_GAS)
        D_elastic = @. h2_self_diffusivity(T_gas_eV) * NIST_H2_N_REF / n_gas
        inv_D = @. 1 / D_elastic + M_H2_GAS * ν_iz / (T_gas_eV * EE) +
            ifelse(vm_g > 0, 3 / (vm_g * L_char), oftype(vm_g, Inf))
        D = @. 1 / inv_D
        λ = @. ifelse(vm_g > 0, 2 * D / vm_g, zero(vm_g))
        return (
            v_para = vm_g, λ_para = λ, v_perp = vm_g, λ_perp = λ,
            vm_para = vm_g, vm_perp = vm_g,
        )
    end

    # Matrices, not vectors: `DiffusionChannel` asserts it lives on the NR×NZ grid.
    n_gas = [0.0 1.0e18; 1.0e20 1.0e22]
    ν_iz = [0.0 1.0e3; 1.0e5 0.0]
    for T in (0.026, 0.5), L in (0.5, Inf)
        ch = neutral_gas_channel(n_gas, T, ν_iz, L)
        ref = oracle_channel(n_gas, T, ν_iz, L)
        for fld in (:v_para, :λ_para, :v_perp, :λ_perp, :vm_para, :vm_perp)
            @test all(isequal.(getfield(ch, fld), getproperty(ref, fld)))
        end
    end
end

@testitem "Ceiling composition: each reads D0, and the tighter answer wins" begin
    using RAPID2D: flux_limited_diffusivity

    # THE DISCRIMINATOR. Both compositions return a finite, plausible number; only the
    # value tells them apart, so a test asserting merely "finite and below D0" passes on
    # either. These numbers are chosen to separate them by 20 %.
    D0, D_wall, S = 10.0, 6.0, 8.0
    F₂(D, s) = D * s / sqrt(D^2 + s^2)

    adopted = min(min(D0, D_wall), F₂(D0, S))
    sequential = F₂(min(D0, D_wall), S)          # retracted; see the design doc §F
    @test adopted ≈ 6.0
    @test sequential ≈ 4.8
    @test !isapprox(adopted, sequential; rtol = 0.01)

    # Why 6 is right: the wall forces D ≤ 6, and at D = 6 the flux is 6|∇n| against a
    # cap of 8|∇n| — the flux bound has room to spare, so there is nothing left to take
    # off. Sequential's 4.8 is suppression with no mechanism behind it.
    @test adopted <= D_wall
    @test adopted <= S

    # The two ceilings are the SAME free-streaming limit at two different lengths, so
    # when both bind the answer is the shorter length, not their quadrature. At D0 → ∞
    # the adopted form gives exactly that and the sequential one gives min/√2.
    big = 1.0e12
    @test isapprox(min(min(big, D_wall), F₂(big, S)), min(D_wall, S); rtol = 1.0e-6)
    @test isapprox(F₂(D_wall, D_wall), D_wall / sqrt(2); rtol = 1.0e-9)

    # Inertness: where geometry does not bind, today's value survives bit for bit.
    @test min(min(D0, Inf), F₂(D0, S)) === F₂(D0, S)
    # ...and the production helper agrees with the hand-written F₂ it is checked against.
    @test flux_limited_diffusivity(D0, 1.0, 8.0 / 0.25, 1.0, 0.25) ≈ F₂(D0, S)
end

@testsnippet CeilingRP begin
    using RAPID2D: SimulationConfig, SimulationFlags, RAPID, initialize!,
        update_transport_quantities!, ion_transport_channels, wall_step_ceiling,
        maxwellian_mean_speed, bulk_ion_mass, flux_limited_diffusivity,
        calculate_para_grad_of_scalar_F, EE

    "A startup-like device with open field lines everywhere in the vessel."
    function ceiling_RP(; NR = 24, NZ = 32, Dpara0 = 0.0, limit_flux = true)
        config = SimulationConfig{Float64}(
            NR = NR, NZ = NZ, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-6, t_end_s = 1.0e-5, R0B0 = 1.0,
            Dpara0 = Dpara0, Dperp0 = 0.0, prefilled_gas_pressure = 5.0e-3,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)   # the writer outlives an auto-cleaned dir
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(
            Ampere = false, Gas_evolve = false, Damp_Transp_outWall = false,
            limit_flux = (state = limit_flux, factor = 0.25),
        )
        initialize!(RP)
        RP.plasma.ne .= 1.0e17
        RP.plasma.ni .= 1.0e17
        RP.plasma.Te_eV .= 5.0
        RP.plasma.Ti_eV .= 1.0
        return RP
    end

    "D∥ of the collisional ion channel, read back out of the channel basis (D = ½vλ)."
    ion_Dpara(RP, species) = begin
        ch = ion_transport_channels(RP, species, nothing)[1]
        @. 0.5 * ch.v_para * ch.λ_para
    end
end

@testitem "Ion ceiling: a collisionless wall-connected cell is exactly D_wall" setup = [CeilingRP] begin
    RP = ceiling_RP()
    tp, pla = RP.transport, RP.plasma
    species = tp.ion_species[1]

    # Collisionless, but WARM. This is the case the whole ceiling exists for, and the
    # arithmetic it used to run through mapped it to D = 0 — the opposite answer.
    tp.νi_neutral .= 0.0
    tp.νi_coulomb .= 0.0

    vm_s = maxwellian_mean_speed.(pla.Ti_eV, species.mass)
    D_wall = @. wall_step_ceiling(vm_s, RP.flf.Lc_forward, RP.flf.Lc_backward)
    D = ion_Dpara(RP, species)

    inw = RP.G.nodes.in_wall_nids
    @test all(n -> isfinite(D[n]), inw)
    @test all(n -> D[n] > 0, inw)
    @test all(n -> isapprox(D[n], D_wall[n]; rtol = 1.0e-10), inw)
end

@testitem "Ion ceiling: cold is zero, and a huge floor is still bounded" setup = [CeilingRP] begin
    RP = ceiling_RP()
    species = RP.transport.ion_species[1]
    inw = RP.G.nodes.in_wall_nids

    # A cold species does not diffuse — and must not produce NaN on the way there, which
    # is what `0/0` in the Einstein form used to do.
    RP.plasma.Ti_eV .= 0.0
    RP.transport.νi_neutral .= 0.0
    RP.transport.νi_coulomb .= 0.0
    D = ion_Dpara(RP, species)
    @test all(n -> D[n] === 0.0, inw)
    @test !any(isnan, D)

    # The floor is added BEFORE the cap, so it cannot transport past the wall either.
    # This is the case that decides whether the ceiling sits on the total or inside the
    # channel sum; placed inside, `Dpara0` would escape it entirely.
    RP2 = ceiling_RP(; Dpara0 = 1.0e12)
    sp2 = RP2.transport.ion_species[1]
    vm_s = maxwellian_mean_speed.(RP2.plasma.Ti_eV, sp2.mass)
    D_wall = @. wall_step_ceiling(vm_s, RP2.flf.Lc_forward, RP2.flf.Lc_backward)
    D2 = ion_Dpara(RP2, sp2)
    @test all(n -> D2[n] <= D_wall[n] * (1 + 1.0e-10), RP2.G.nodes.in_wall_nids)
    @test all(n -> isapprox(D2[n], D_wall[n]; rtol = 1.0e-10), RP2.G.nodes.in_wall_nids)
end

@testitem "Electron ceilings: the production path takes the tighter of two answers" setup = [CeilingRP] begin
    # A large floor puts `D0` far above both ceilings, so each one's answer is its own
    # scale rather than a lightly-corrected `D0`.
    RP = ceiling_RP(; Dpara0 = 1.0e12)
    tp, pla = RP.transport, RP.plasma
    me = RP.config.constants.me
    inw = RP.G.nodes.in_wall_nids

    # A PARALLEL DENSITY GRADIENT IS REQUIRED HERE. With `ne` uniform, Lₙ = ∞ and the
    # flux closure is the identity — the adopted and withdrawn compositions would then
    # agree everywhere and this test would pass while proving nothing.
    @. pla.ne = 1.0e17 * (2 + RP.G.Z2D)
    @. pla.ni = pla.ne

    vm_e = maxwellian_mean_speed.(pla.Te_eV, me)
    ∇para_ne = calculate_para_grad_of_scalar_F(RP, pla.ne; upwind = false)
    # The flux closure's own scale, S = ¼v̄n/|∇∥n|.
    S = @. 0.25 * vm_e * pla.ne / abs(∇para_ne)

    # Put the wall EXACTLY at the flux scale: `wall_step_ceiling` with Lf = Lb = 2S/v̄
    # returns ¼v̄(2·2S/v̄) = S. That is the crossover, and the crossover is where the two
    # compositions are furthest apart — the withdrawn sequential form returns S/√2 there,
    # 29.3 % low, because it combines two statements of one free-streaming limit in
    # quadrature.
    @. RP.flf.Lc_forward = 2 * S / vm_e
    @. RP.flf.Lc_backward = 2 * S / vm_e
    update_transport_quantities!(RP)

    D0 = @. tp.Dpara0 + tp.Dpara_e_eff
    D_geom = @. min(D0, wall_step_ceiling(vm_e, RP.flf.Lc_forward, RP.flf.Lc_backward))
    D_flux = @. flux_limited_diffusivity(D0, ∇para_ne, pla.ne, vm_e, 0.25)

    adopted = @. min(D_geom, D_flux)
    sequential = @. flux_limited_diffusivity(D_geom, ∇para_ne, pla.ne, vm_e, 0.25)

    @test all(n -> isapprox(tp.Dpara[n], adopted[n]; rtol = 1.0e-12), inw)
    # Both guards below exist so the assertion above cannot pass vacuously.
    @test any(n -> D_geom[n] < D0[n] * (1 - 1.0e-9), inw)
    @test any(n -> !isapprox(sequential[n], adopted[n]; rtol = 1.0e-6), inw)
    # And the disagreement is the predicted 1/√2, not an arbitrary difference.
    k = argmax([isfinite(S[n]) ? -abs(D_geom[n] / S[n] - 1) : -Inf for n in inw])
    node = inw[k]
    @test isapprox(sequential[node] / adopted[node], 1 / sqrt(2); rtol = 5.0e-2)
end

@testitem "Electron ceilings: an open device at breakdown is untouched" setup = [CeilingRP] begin
    RP = ceiling_RP()
    tp = RP.transport
    inw = RP.G.nodes.in_wall_nids

    update_transport_quantities!(RP)
    with_geometry = copy(tp.Dpara)

    # Same state, but with no wall in either direction: the geometric ceiling is absent
    # by construction. At ordinary breakdown parameters the two must agree exactly —
    # `D_wall` is ~10⁶ against a `D0` of ~10³, so `min` is a no-op and the flux closure
    # sees precisely the input it saw before this feature existed.
    RP.flf.Lc_forward .= Inf
    RP.flf.Lc_backward .= Inf
    update_transport_quantities!(RP)

    @test all(n -> with_geometry[n] === tp.Dpara[n], inw)
    @test all(n -> isfinite(tp.Dpara[n]), inw)
end

@testitem "Ion channel gate: the diagnostic names the nodes, whichever field diverges" setup = [CeilingRP] begin
    using RAPID2D: DiffusionChannel

    RP = ceiling_RP()
    NR, NZ = RP.G.NR, RP.G.NZ

    # A turbulent channel whose BOOKKEEPING SPEED diverges — E×B with B_tot → 0 does
    # exactly this. The finiteness gate checks all six channel fields, so it fires; the
    # diagnostic used to scan only three of them, so this precise failure reported
    # "not finite at 0 node(s)" with no coordinates to act on. The gate and the report
    # must never disagree about what "non-finite" means.
    z = zeros(NR, NZ)
    v = fill(1.0, NR, NZ)
    bad_vm = fill(1.0, NR, NZ)
    bad_vm[5, 7] = Inf
    turb = DiffusionChannel(v, z, v, z; vm_para = v, vm_perp = bad_vm)

    err = try
        RAPID2D.ion_transport_channels(RP, RP.transport.ion_species[1], turb)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("turbulent", err.msg)
    @test occursin("1 node(s)", err.msg)
    # The coordinates are the actionable part: without them the error names a species
    # and a mechanism but not a place to look.
    @test occursin("(R,Z)=", err.msg)
    @test occursin(string(round(RP.G.R1D[5], digits = 4)), err.msg)
end
