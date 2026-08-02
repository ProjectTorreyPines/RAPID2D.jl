# Kinetic diffusivity of the neutral H2 fill.
#
# The mean free path competes three ways: gas-gas elastic collisions, destruction
# by electron-impact ionization, and — once the gas is thin enough that a molecule
# would otherwise cross the vessel unimpeded — the wall itself. All three are
# summed as rates (Matthiessen), so whichever is shortest wins.

@testitem "Neutral gas diffusivity: elastic mean free path scales as 1/n" begin
    using RAPID2D: neutral_gas_diffusivity

    # D = ½·v_th·λ at fixed T, so D inherits λ's density scaling exactly.
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

@testitem "Neutral gas diffusivity: ionization shortens the mean free path" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # A molecule destroyed by electron impact never completes its free path, so
    # ν_iz adds to the collision rate. This is the mechanism behind neutral
    # shielding: where the plasma is hot, the gas cannot penetrate.
    n, T, νiz = 6.4e17, 0.026, 1.0e4
    D0 = neutral_gas_diffusivity(n, T, 0.0, Inf)
    Diz = neutral_gas_diffusivity(n, T, νiz, Inf)
    @test Diz < D0

    # exact combination law, without hard-coding σ: recover ν_elastic from D0
    vth = neutral_gas_thermal_speed(T)
    ν_el = vth^2 / (2 * D0)
    @test Diz ≈ D0 * ν_el / (ν_el + νiz) rtol = 1.0e-12
end

@testitem "Neutral gas diffusivity: never transports faster than free streaming" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # At breakdown pressures the gas-gas mean free path is METRES — larger than the
    # vessel — so the flow is free-molecular, not diffusive. Left uncapped, D grows
    # without bound and the diffusive crossing time L²/D drops BELOW the ballistic
    # floor L/v_th, which nothing can beat. Adding the wall as a collision channel
    # (1/λ += 1/L, i.e. Knudsen diffusion) restores the correct limit.
    L, T = 1.09, 0.026
    vth = neutral_gas_thermal_speed(T)
    for n in (1.0e15, 1.0e17, 6.4e17, 1.0e19)
        D = neutral_gas_diffusivity(n, T, 0.0, L)
        @test L^2 / D ≥ L / vth
    end

    # in the collisionless limit the wall alone sets the path
    @test neutral_gas_diffusivity(1.0e8, T, 0.0, L) ≈ 0.5 * vth * L rtol = 1.0e-4
end

@testitem "Neutral gas diffusivity: stays finite where the gas is fully burnt" begin
    using RAPID2D: neutral_gas_diffusivity, neutral_gas_thermal_speed

    # n_gas → 0 makes the elastic path infinite. The wall term keeps λ finite, so
    # no special-casing (and no NaN scrubbing) is needed on burnt-out cells.
    L, T = 1.09, 0.026
    D = neutral_gas_diffusivity(0.0, T, 0.0, L)
    @test isfinite(D)
    @test D ≈ 0.5 * neutral_gas_thermal_speed(T) * L rtol = 1.0e-12
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

    n_before = copy(RP.plasma.n_H2_gas)
    update_neutral_H2_gas_density!(RP)

    inw = RP.G.nodes.in_wall_nids
    expected = @. n_before - RP.dt * RP.plasma.ne * RP.plasma.ν_en_iz
    @test RP.plasma.n_H2_gas[inw] ≈ expected[inw] rtol = 1.0e-10
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
