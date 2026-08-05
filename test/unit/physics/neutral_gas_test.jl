# Kinetic diffusivity of the neutral H2 fill.
#
# The mean free path competes three ways: gas-gas elastic collisions, destruction
# by electron-impact ionization, and — once the gas is thin enough that a molecule
# would otherwise cross the vessel unimpeded — the wall itself. All three are
# summed as rates (Matthiessen), so whichever is shortest wins.

@testitem "Neutral gas diffusivity: elastic mean free path scales as 1/n" begin
    using RAPID2D: neutral_gas_diffusivity

    # D = ½·vth_g·λ at fixed T, so D inherits λ's density scaling exactly.
    # Inf characteristic length disables the wall term to isolate the gas term.
    D1 = neutral_gas_diffusivity(1.0e18, 0.026, 0.0, Inf)
    D2 = neutral_gas_diffusivity(5.0e17, 0.026, 0.0, Inf)
    @test D2 / D1 ≈ 2.0 rtol = 1.0e-12
end

@testitem "Neutral gas diffusivity: reproduces the NIST reference value" begin
    using RAPID2D: neutral_gas_diffusivity, NIST_H2_T_REF_K, NIST_H2_N_REF

    # NIST TN 2279 (Burgess 2024), Table 1a "Small Molecules", Hydrogen-Oxygen:
    #   H2 in H2, T_range 115-295 K, T_ref 298 K, D_ref = 1.309 cm²/s
    # at 101.325 kPa. Pinning the absolute value is what stops the MATLAB's
    # 0.61e-4 m²/s from being transcribed in: that number is the H2-in-AIR binary
    # coefficient, a factor 2.1 below self-diffusion, and no scaling law would
    # reveal the swap.
    T_ref_eV = NIST_H2_T_REF_K * 8.617333262e-5
    D = neutral_gas_diffusivity(NIST_H2_N_REF, T_ref_eV, 0.0, Inf)
    @test D ≈ 1.309e-4 rtol = 1.0e-3
end

@testitem "Neutral gas diffusivity: follows the NIST temperature fit, not sqrt(T)" begin
    using RAPID2D: neutral_gas_diffusivity, NIST_H2_N_REF

    # ln(D) = A + B/T + C·ln(T) with C = 1.686 means D ∝ T^1.69, the usual real-gas
    # behaviour once the attractive potential is included. A hard-sphere mean free
    # path would give √T and land 2.5× low over this span, so this assertion
    # separates "calibrated to NIST" from "hard sphere normalised at one point".
    kB_eV = 8.617333262e-5
    Ta, Tb = 150.0, 290.0                       # inside the fit's 115-295 K range
    Da = neutral_gas_diffusivity(NIST_H2_N_REF, Ta * kB_eV, 0.0, Inf)
    Db = neutral_gas_diffusivity(NIST_H2_N_REF, Tb * kB_eV, 0.0, Inf)

    A, B, C = -9.309, -8.028, 1.686
    nist(T) = 1.0e-4 * exp(A + B / T + C * log(T))
    @test Db / Da ≈ nist(Tb) / nist(Ta) rtol = 1.0e-6
    @test Db / Da > 1.5 * sqrt(Tb / Ta)         # decisively not the √T law
end

@testitem "Neutral gas diffusivity: each mechanism is exact alone, and they compose at D" begin
    using RAPID2D: neutral_gas_diffusivity, h2_self_diffusivity, NIST_H2_N_REF,
        maxwellian_mean_speed, M_H2_GAS, EE_GAS

    # Three mechanisms bound a molecule's path, and each has an exact closed form
    # when it acts alone. Isolating one means switching the other two off through
    # their own null values, so every branch is exercised:
    #
    #   elastic only   D = D_NIST(T)·n_ref/n     the measurement, unmodified
    #   ionization     D = T/(m·ν_iz)            Einstein — ν_iz is a RATE
    #   wall only      D = ⅓·vm·L                Knudsen, D_K = ⅓v̄d (Kennard §64)
    #
    # and the model is that they compose as diffusivities. The `⅓` is the isotropic
    # projection ⟨v_z²/|v|⟩ = vm/3; a magnetized ion would get ½ instead.
    #
    # ν_iz belongs in D at all because absorption damps the CURRENT as well as the
    # density — the same reason neutron diffusion theory puts Σ_a in the transport
    # cross section — so this is not double counting against the burn-out sink.
    n, T, νiz, L = 6.4e17, 0.026, 1.0e4, 1.09
    vm_g = maxwellian_mean_speed(T, M_H2_GAS)

    @test neutral_gas_diffusivity(n, T, 0.0, Inf) ≈
        h2_self_diffusivity(T) * NIST_H2_N_REF / n rtol = 1.0e-14
    @test neutral_gas_diffusivity(0.0, T, νiz, Inf) ≈
        T * EE_GAS / (M_H2_GAS * νiz) rtol = 1.0e-14
    @test neutral_gas_diffusivity(0.0, T, 0.0, L) ≈ vm_g * L / 3 rtol = 1.0e-14

    @test 1 / neutral_gas_diffusivity(n, T, νiz, L) ≈
        1 / neutral_gas_diffusivity(n, T, 0.0, Inf) +
        1 / neutral_gas_diffusivity(0.0, T, νiz, Inf) +
        1 / neutral_gas_diffusivity(0.0, T, 0.0, L) rtol = 1.0e-14

    # Adding a loss channel can only shorten the path — the physical content of a
    # sum of inverse diffusivities, and the mechanism behind neutral shielding:
    # where the plasma is hot, the gas cannot penetrate.
    @test neutral_gas_diffusivity(n, T, νiz, L) < neutral_gas_diffusivity(n, T, 0.0, L)
end

@testitem "Neutral gas diffusivity: never transports faster than free streaming" begin
    using RAPID2D: neutral_gas_diffusivity, maxwellian_thermal_speed,
        maxwellian_mean_speed, M_H2_GAS

    # At breakdown pressures the gas-gas mean free path is METRES — larger than the
    # vessel — so the flow is free-molecular, not diffusive. Left uncapped, D grows
    # without bound and the diffusive crossing time L²/D drops BELOW the ballistic
    # floor L/vth_g, which nothing can beat. Adding the wall as a collision channel
    # (1/λ += 1/L, i.e. Knudsen diffusion) restores the correct limit.
    L, T = 1.09, 0.026
    vth_g = maxwellian_thermal_speed(T, M_H2_GAS)
    for n in (1.0e15, 1.0e17, 6.4e17, 1.0e19)
        D = neutral_gas_diffusivity(n, T, 0.0, L)
        @test L^2 / D ≥ L / vth_g
    end

    # and the bound is approached from below as the gas thins towards free-molecular
    @test neutral_gas_diffusivity(1.0e8, T, 0.0, L) < maxwellian_mean_speed(T, M_H2_GAS) * L / 3
end

@testitem "Neutral gas diffusivity: stays finite where the gas is fully burnt" begin
    using RAPID2D: neutral_gas_diffusivity, maxwellian_mean_speed, M_H2_GAS

    # n_gas → 0 makes the elastic path infinite. The wall term keeps λ finite, so
    # no special-casing (and no NaN scrubbing) is needed on burnt-out cells.
    L, T = 1.09, 0.026
    D = neutral_gas_diffusivity(0.0, T, 0.0, L)
    @test isfinite(D)
    @test D ≈ maxwellian_mean_speed(T, M_H2_GAS) * L / 3 rtol = 1.0e-12
end

# ── reflective diffusion operator ───────────────────────────────────────────
# The shared ∇𝐃∇ builder is not reusable here: it sweeps every interior node
# without wall awareness, so zeroing D outside the wall still leaves a coupling
# coefficient inv_J·½·CT_in to the outside neighbour and the gas leaks out. A
# reflective wall must OMIT the outside neighbour, which is a different stencil.

@testitem "Reflective diffusion operator: rows sum to zero" begin
    using RAPID2D: build_reflective_diffusion_matrix
    using RAPID2D.SparseArrays

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    D = fill(50.0, RP.G.NR, RP.G.NZ)
    A = build_reflective_diffusion_matrix(RP.G, D)

    # zero row sum is what makes the stencil a divergence: no node manufactures
    # or destroys gas on its own. The residual is pure cancellation round-off, so
    # it must be judged against the size of the coefficients being cancelled —
    # an absolute bound would silently track the grid spacing and D.
    @test maximum(abs, sum(A, dims = 2)) < 1.0e-13 * maximum(abs, A)
end

@testitem "Reflective diffusion operator: uniform gas produces no flux" begin
    using RAPID2D: build_reflective_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # spatially varying D, so a passing test cannot be an artefact of constant D
    D = [
        30.0 + 20.0 * sin(3.0 * RP.G.R1D[i]) * cos(2.0 * RP.G.Z1D[j])
            for i in 1:RP.G.NR, j in 1:RP.G.NZ
    ]
    A = build_reflective_diffusion_matrix(RP.G, D)

    # A·const is the row sum times n, so the tolerance has to carry BOTH factors.
    # A genuine leak (an absorbing wall) lands near 0.25 in these relative units,
    # thirteen orders above the bound below, so this cannot pass by accident.
    n_val = 7.0e18
    n_uniform = fill(n_val, RP.G.NR * RP.G.NZ)
    @test maximum(abs, A * n_uniform) < 1.0e-13 * maximum(abs, A) * n_val
end

@testitem "Reflective diffusion operator: no coupling across the wall" begin
    using RAPID2D: build_reflective_diffusion_matrix
    using RAPID2D.SparseArrays

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    D = fill(50.0, RP.G.NR, RP.G.NZ)
    A = build_reflective_diffusion_matrix(RP.G, D)

    # THE defining property. An absorbing wall would show non-zero entries here,
    # and the gas would drain into nodes nothing ever solves for.
    outside = RP.G.nodes.on_out_wall_nids
    inside = RP.G.nodes.in_wall_nids
    @test maximum(abs, A[inside, outside]) == 0.0

    # and nothing outside the wall evolves at all
    @test maximum(abs, A[outside, :]) == 0.0
end

@testitem "Reflective diffusion operator: conserves Jacobian-weighted particles" begin
    using RAPID2D: build_reflective_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    D = [
        30.0 + 20.0 * sin(3.0 * RP.G.R1D[i]) * cos(2.0 * RP.G.Z1D[j])
            for i in 1:RP.G.NR, j in 1:RP.G.NZ
    ]
    A = build_reflective_diffusion_matrix(RP.G, D)

    # In cylindrical geometry the conserved quantity is Σ J·n, not Σ n — the same
    # invariant the impurity wall ledger uses. d/dt(Σ J n) = (Jᵀ A) n must vanish
    # for EVERY n, i.e. the Jacobian-weighted column sums are zero.
    Jv = vec(RP.G.Jacob)
    colsum = vec(Jv' * A)
    @test maximum(abs, colsum[RP.G.nodes.in_wall_nids]) < 1.0e-8 * maximum(Jv)
end

@testitem "Reflective diffusion operator: smooths a peak" begin
    using RAPID2D: build_reflective_diffusion_matrix

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    D = fill(50.0, RP.G.NR, RP.G.NZ)
    A = build_reflective_diffusion_matrix(RP.G, D)

    Rc = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    Zc = (RP.G.Z1D[1] + RP.G.Z1D[end]) / 2
    n = [
        exp(-((RP.G.R2D[i, j] - Rc)^2 + (RP.G.Z2D[i, j] - Zc)^2) / 0.02)
            for i in 1:RP.G.NR, j in 1:RP.G.NZ
    ]
    rhs = reshape(A * vec(n), RP.G.NR, RP.G.NZ)

    # diffusion drains the peak into its surroundings
    ipk, jpk = Tuple(argmax(n))
    @test rhs[ipk, jpk] < 0
end

# ── the update itself ───────────────────────────────────────────────────────

@testitem "Neutral gas update: pure diffusion conserves Jacobian-weighted gas" begin
    using RAPID2D: update_neutral_H2_gas_density!

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-5,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    RP.plasma.ne .= 0.0                     # no ionization sink: diffusion alone
    RAPID2D.update_transport_quantities!(RP)

    Rc = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    Zc = (RP.G.Z1D[1] + RP.G.Z1D[end]) / 2
    @. RP.plasma.n_H2_gas = 1.0e18 *
        (1.0 + 3.0 * exp(-((RP.G.R2D - Rc)^2 + (RP.G.Z2D - Zc)^2) / 0.02))

    inw = RP.G.nodes.in_wall_nids
    Jv = vec(RP.G.Jacob)
    total(x) = sum(Jv[k] * x[k] for k in inw)
    before = total(RP.plasma.n_H2_gas)

    # The sink consumes rates the electron solve publishes, so a standalone call
    # has to establish them. Nothing is ionizing here, which is the point: this
    # test is about the diffusion stencil alone.
    RAPID2D.update_reaction_counts!(RP)

    for _ in 1:20
        update_neutral_H2_gas_density!(RP)
    end

    # Σ J·n is the invariant of the reflective stencil; plain Σ n is not conserved
    # in cylindrical geometry. Twenty steps so any per-step leak accumulates.
    @test total(RP.plasma.n_H2_gas) ≈ before rtol = 1.0e-10
    @test !(RP.plasma.n_H2_gas ≈ fill(RP.plasma.n_H2_gas[inw[1]], size(RP.plasma.n_H2_gas)))
end

@testitem "Neutral gas update: the sink is exactly the electron source" begin
    using RAPID2D: update_neutral_H2_gas_density!

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 25, NZ = 30,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # One electron born per molecule destroyed. The gas must lose exactly what
    # solve_electron_continuity_equation! adds, i.e. dt·ne·ν_en_iz with the SAME
    # ν the electron equation reads — otherwise nuclei are created or destroyed.
    # A uniform state makes diffusion a no-op, isolating the sink arithmetic.
    RP.plasma.ne .= 1.0e18
    RAPID2D.update_transport_quantities!(RP)
    RP.plasma.ν_en_iz .= 5.0e5

    # Run the producer, then the sink — the order `advance_timestep!` uses. The
    # sink must destroy exactly what the electron equation created, so `expected`
    # is read off the published rate rather than rebuilt from `ne·ν_iz`: that
    # rebuild is what used to differ from the electron equation at θ < 1.
    ne_before = copy(RP.plasma.ne)
    RAPID2D.solve_electron_continuity_equation!(RP)
    n_before = copy(RP.plasma.n_H2_gas)
    update_neutral_H2_gas_density!(RP)

    inw = RP.G.nodes.in_wall_nids
    expected = @. n_before - RP.reactions.counts.iz
    @test RP.plasma.n_H2_gas[inw] ≈ expected[inw] rtol = 1.0e-10
    # and that IS the electron gain, one nucleus each way
    @test (RP.plasma.ne - ne_before)[inw] ≈ (n_before - RP.plasma.n_H2_gas)[inw] rtol = 1.0e-9
end

@testitem "Neutral gas update: density never goes negative" begin
    using RAPID2D: update_neutral_H2_gas_density!

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 20, NZ = 24,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-3,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # a step long enough that an unguarded explicit sink would overshoot into
    # negative density, which would then poison every ν = n_H2·K downstream
    RP.plasma.ne .= 1.0e19
    RAPID2D.update_transport_quantities!(RP)
    RP.plasma.ν_en_iz .= 1.0e6

    RAPID2D.update_reaction_counts!(RP)
    update_neutral_H2_gas_density!(RP)
    @test minimum(RP.plasma.n_H2_gas) ≥ 0.0
end

@testitem "Neutral gas update: the Gas_evolve path is alive" begin
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 20, NZ = 24,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-7,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.plasma.ne .= 1.0e18
    RP.plasma.Te_eV .= 20.0
    RAPID2D.update_transport_quantities!(RP)

    # This branch used to call a function that did not exist anywhere in src, so
    # advance_timestep! threw UndefVarError with the DEFAULT flag set.
    RP.flags.Gas_evolve = true
    before = copy(RP.plasma.n_H2_gas)
    RAPID2D.advance_timestep!(RP)
    @test RP.plasma.n_H2_gas != before

    RP.flags.Gas_evolve = false
    frozen = copy(RP.plasma.n_H2_gas)
    RAPID2D.advance_timestep!(RP)
    @test RP.plasma.n_H2_gas == frozen
end

@testitem "Neutral gas update: theta_gas selects the time scheme, BE by default" begin
    using RAPID2D: update_neutral_H2_gas_density!

    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 20, NZ = 24,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-5,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # Backward Euler by default, and deliberately its own member of θ_imp rather
    # than the transport weight: that one is 0.5, which is Crank-Nicolson. CN is
    # A-stable but not L-stable, so its amplification factor tends to −1 as |λ|Δt
    # grows and stiff modes ring instead of damping. D spans two orders of
    # magnitude across the shielding layer, so the stiff end is always present here.
    @test RP.flags.θ_imp.gas == 1.0
    @test RP.flags.θ_imp.transport == 0.5

    # the knob is honoured: θ = 0 is forward Euler, which on this operator must
    # blow up at a Δt far above the explicit CFL limit min(dR,dZ)²/(4D)
    RP.plasma.ne .= 0.0
    RAPID2D.update_transport_quantities!(RP)
    Rc = (RP.G.R1D[1] + RP.G.R1D[end]) / 2
    Zc = (RP.G.Z1D[1] + RP.G.Z1D[end]) / 2
    bump() = @. 1.0e18 * (
        1.0 + 3.0 *
            exp(-((RP.G.R2D - Rc)^2 + (RP.G.Z2D - Zc)^2) / 0.02)
    )

    RAPID2D.update_reaction_counts!(RP)   # no ionization here; the scheme is under test

    RP.plasma.n_H2_gas .= bump()
    RP.flags.θ_imp.gas = 1.0
    for _ in 1:20
        update_neutral_H2_gas_density!(RP)
    end
    @test all(isfinite, RP.plasma.n_H2_gas)
    @test minimum(RP.plasma.n_H2_gas) ≥ 0.0

    RP.plasma.n_H2_gas .= bump()
    RP.flags.θ_imp.gas = 0.0
    for _ in 1:20
        update_neutral_H2_gas_density!(RP)
    end
    @test !all(isfinite, RP.plasma.n_H2_gas) ||
        minimum(RP.plasma.n_H2_gas) < 0.0
end
