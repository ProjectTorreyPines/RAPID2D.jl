# Physics module tests. One @testitem per PHYSICS SCENARIO, self-contained and readable
# top to bottom: grid config, SimulationFlags, initial conditions, the run, the
# assertions. Numerical-scheme variations of the SAME scenario are nested @testsets
# inside their scenario's item.
#
# Some scenarios are STRICTLY SEQUENTIAL (they chain state with no reset between
# blocks); each is marked with a SEQUENTIAL comment explaining why it cannot be split.

@testsnippet PhysicsFixtures begin
    # Gaussian density blob on the grid. Pure geometry — the parameters that matter to
    # a given test (centre, widths, peak) are always passed at the call site.
    function gaussian_density(G; R0, Z0, σR, σZ, peak)
        return @. peak * exp(-((G.R2D - R0)^2 / (2σR^2) + (G.Z2D - Z0)^2 / (2σZ^2)))
    end

    # Snapshot writers resolve Output_path relative to the process cwd, and
    # TestItemRunner cd's into each test file's directory. cleanup=false is REQUIRED:
    # the RAPID constructor opens ADIOS handles here (src/types.jl) that are closed by a
    # FINALIZER at a GC-determined time, so the directory must outlive the RAPID object.
    # A self-deleting tempdir aborts the process with "Bad file descriptor".
    scratch_output_dir() = mktempdir(; cleanup = false)
end

# ── Initialization ───────────────────────────────────────────────────────────────────

@testitem "Physics: module initialization basics" begin
    # A freshly initialized RAPID object in the default "manual" device geometry:
    # uniform seed density inside the wall, room-temperature plasma, purely vertical
    # external field, and no flow.
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 40, NZ = 80,
        prefilled_gas_pressure = 1.0e-2,   # Pa
        R0B0 = 1.0,                        # T·m
        dt = 1.0e-8,
        snap0D_Δt_s = 1.0e-7,
        snap2D_Δt_s = 1.0e-6,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # Densities: seeded inside the wall, zero on/outside it, and quasi-neutral
    @test all(RP.plasma.ne[RP.G.nodes.in_wall_nids] .== 1.0e6)
    @test all(RP.plasma.ne[RP.G.nodes.on_out_wall_nids] .== 0.0)
    @test RP.plasma.ne == RP.plasma.ni

    # Temperatures start at room temperature
    @test all(RP.plasma.Te_eV .≈ RP.config.constants.room_T_eV)
    @test all(RP.plasma.Ti_eV .≈ RP.config.constants.room_T_eV)

    # Field unit vectors are normalised; manual setup gives BR = 0, BZ > 0
    @test all(isapprox.(RP.fields.bR .^ 2 + RP.fields.bZ .^ 2 + RP.fields.bϕ .^ 2, 1.0, atol = 1.0e-10))
    @test all(RP.fields.BR .== 0.0)
    @test all(RP.fields.BZ .> 0.0)

    # With no self-field yet, the total parallel E equals the external one
    @test all(RP.fields.E_para_tot .== RP.fields.E_para_ext)
    @test all(RP.fields.E_para_ext .== RP.fields.Eϕ_ext .* RP.fields.bϕ)

    # No initial flow
    @test all(RP.plasma.ue_para .== 0.0)
    @test all(RP.plasma.ui_para .== 0.0)
end

@testitem "Physics: reaction rate coefficient lookups" begin
    # Smallest possible grid: this only checks that the RRC interpolators return
    # correctly shaped, non-negative arrays for both electrons and H2 ions.
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 10, NZ = 10,
        prefilled_gas_pressure = 1.0e-2,
        R0B0 = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)

    RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
    RRC_mom_tot = get_electron_RRC(RP, RP.eRRCs, :Total_Momentum)
    @test size(RRC_iz) == (RP.G.NR, RP.G.NZ)
    @test size(RRC_mom_tot) == (RP.G.NR, RP.G.NZ)
    @test all(RRC_iz .>= 0.0)
    @test all(RRC_mom_tot .>= 0.0)

    iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
    iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)
    @test size(iRRC_elastic) == (RP.G.NR, RP.G.NZ)
    @test size(iRRC_cx) == (RP.G.NR, RP.G.NZ)
    @test all(iRRC_elastic .>= 0.0)
    @test all(iRRC_cx .>= 0.0)
end

# SEQUENTIAL — do not split. The ue_para golden depends on the 100-iteration warm-up
# loop that precedes it, and the final diffusion check depends on the Gaussian ne
# overwrite that precedes THAT. Order is load-bearing throughout.
@testitem "Physics: density transport RHS terms" begin
    using RAPID2D.Statistics

    # Explicit scheme with diffusion, convection and ionization all ON — this checks the
    # individual RHS operators rather than an end-to-end evolution.
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 20, NZ = 20,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 1.0e-6,
        Dperp0 = 0.1,
    )
    RP = RAPID{Float64}(config)
    RP.flags.Implicit = false      # explicit, to inspect the raw RHS terms
    RP.flags.diffu = true
    RP.flags.convec = true
    RP.flags.src = true
    RP.flags.Include_ud_convec_term = false
    RP.flags.Include_ud_pressure_term = false
    RP.flags.Ionz_method = "Xsec"
    initialize!(RP)

    # Drive the parallel velocity to its steady state before measuring it
    RP.plasma.Te_eV .= 10.0
    for _ in 1:100
        update_transport_quantities!(RP)
        update_ue_para!(RP)
    end

    # Golden re-measured against the current RRC table; re-measure if the table changes.
    @test mean(RP.plasma.ue_para[RP.G.nodes.in_wall_nids]) ≈ -492253.1332931324

    op = RP.operators
    update_RRCs!(RP)

    # Ionization source is non-zero inside the wall and zero outside it
    @test !all(RP.plasma.ν_en_iz .== 0.0)
    @test all(RP.plasma.ν_en_iz[RP.G.nodes.on_out_wall_nids] .== 0.0)

    # ne is still uniform inside the wall, so the diffusion term must vanish there —
    # checked both via the direct evaluation and via the assembled operator.
    @test all(compute_∇𝐃∇f_directly(RP, RP.plasma.ne)[RP.G.nodes.inWall_deepInWall_nids] .== 0.0)
    RHS_diffu = (op.∇𝐃∇ * RP.plasma.ne)
    mean_inside_ne = mean(RP.plasma.ne[RP.G.nodes.in_wall_nids])
    @test all(isapprox.(RHS_diffu[RP.G.nodes.inWall_deepInWall_nids], 0.0, atol = 1.0e-12 * mean_inside_ne))

    # Introduce a density gradient; diffusion must now be non-zero
    inside_idx = RP.G.nodes.in_wall_nids
    center = [RP.G.NR ÷ 2, RP.G.NZ ÷ 2]
    for i in inside_idx
        r, z = RP.G.nodes.rid[i], RP.G.nodes.zid[i]
        dist = sqrt((r - center[1])^2 + (z - center[2])^2)
        RP.plasma.ne[i] = 1.0e6 * exp(-dist^2 / 20.0)
    end
    RHS_diffu = (op.∇𝐃∇ * RP.plasma.ne)
    @test !all(RHS_diffu[RP.G.nodes.in_wall_nids] .== 0.0)
end

# ── Transport scenarios ──────────────────────────────────────────────────────────────

@testitem "Pure Convection: constant ue_para" setup = [PhysicsFixtures] begin
    # A Gaussian blob is advected along B at a CONSTANT parallel velocity. Convection is
    # the only transport term enabled — no sources, diffusion, heating, or field
    # evolution — so the density centroid must move by exactly ue_para·b·t_end.
    # Repeated over all four (implicit × upwind) scheme combinations.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 50, NZ = 70,
        R_min = 0.1, R_max = 0.5,
        Z_min = -0.4, Z_max = 0.4,
        dt = 1.0e-6, t_end_s = 100.0e-6,
        R0B0 = 1.0,
        Dpara0 = 10.0, Dperp0 = 0.1,          # irrelevant here: diffu is off
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [0.15, 0.45, 0.45, 0.15],
        wall_Z = [-0.35, -0.35, 0.35, 0.35],
        snap0D_Δt_s = 10.0e-6,
        snap2D_Δt_s = 20.0e-6,
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        convec = true,                    # ← the only transport term under test
        # everything else deliberately off, so the displacement is purely advective
        src = false,
        diffu = false,
        ud_evolve = false,                # ue_para must stay at its initial value
        ud_method = "Xsec",
        Te_evolve = false,
        Ti_evolve = false,
        Ampere = false,
        E_para_self_ES = false,
        E_para_self_EM = false,
        Gas_evolve = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Coulomb_Collision = false,
        negative_n_correction = false,
    )

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    σR = (config.R_max - config.R_min) / 16
    σZ = (config.Z_max - config.Z_min) / 16

    initialize!(RP)
    ini_ne = gaussian_density(RP.G; R0, Z0, σR, σZ, peak = 1.0e6)
    ini_ne[RP.G.nodes.on_out_wall_nids] .= 0.0

    ini_ue_para = 1.0e6        # m/s, held constant
    ini_BR_ext = 10.0e-4       # T — tilts B so the blob moves in both R and Z
    ini_BZ_ext = 20.0e-4

    function reset_to_initial_conditions!(RP)
        RP.plasma.ne = copy(ini_ne)
        RP.plasma.ue_para .= ini_ue_para
        RP.fields.BR_ext .= ini_BR_ext
        RP.fields.BZ_ext .= ini_BZ_ext
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    for implicit in (false, true), upwind in (false, true)
        @testset "Implicit=$implicit, upwind=$upwind" begin
            RP.flags.Implicit = implicit
            RP.flags.upwind = upwind
            initialize!(RP)
            reset_to_initial_conditions!(RP)

            @test all(RP.plasma.ne .>= 0.0)

            # Expected displacement from the constant parallel velocity
            ΔR_2D = @. RP.plasma.ue_para * RP.fields.bR * RP.t_end_s
            ΔZ_2D = @. RP.plasma.ue_para * RP.fields.bZ * RP.t_end_s
            expected_R = sum(ini_ne .* ΔR_2D) / sum(ini_ne)
            expected_Z = sum(ini_ne .* ΔZ_2D) / sum(ini_ne)

            RAPID2D.run_simulation!(RP)

            actual_R = sum(RP.plasma.ne .* RP.G.R2D) / sum(RP.plasma.ne) - R0
            actual_Z = sum(RP.plasma.ne .* RP.G.Z2D) / sum(RP.plasma.ne) - Z0

            # Upwind is positivity-preserving; the central scheme may undershoot slightly
            if upwind
                @test all(RP.plasma.ne .>= 0.0)
            else
                @test all(RP.plasma.ne .>= -1.0e-9 * maximum(ini_ne))
            end

            @test isapprox(actual_R, expected_R, rtol = 5.0e-2)
            @test isapprox(actual_Z, expected_Z, rtol = 5.0e-2)
        end
    end

    @testset "implicit θ=0 ≡ explicit, θ=1 differs" begin
        # upwind is set EXPLICITLY: otherwise it leaks in from the last loop iteration.
        RP.flags.upwind = true

        RP.flags.Implicit = false
        initialize!(RP); reset_to_initial_conditions!(RP)
        RAPID2D.run_simulation!(RP)
        RP_explicit = deepcopy(RP)

        RP.flags.Implicit = true
        # every family, or the comparison against the explicit run is not one:
        # the explicit path is θ = 0 for transport AND for the atomic rates.
        RP.flags.θ_imp.transport = 0.0
        RP.flags.θ_imp.growth = 0.0
        initialize!(RP); reset_to_initial_conditions!(RP)
        RAPID2D.run_simulation!(RP)
        RP_implicit_0 = deepcopy(RP)

        RP.flags.Implicit = true
        RP.flags.θ_imp.transport = 1.0
        RP.flags.θ_imp.growth = 1.0
        initialize!(RP); reset_to_initial_conditions!(RP)
        RAPID2D.run_simulation!(RP)
        RP_implicit_1 = deepcopy(RP)

        # θ=0 is algebraically identical to the explicit update
        @test isapprox(RP_explicit.plasma.ne, RP_implicit_0.plasma.ne, rtol = 1.0e-12)
        # θ=1 is close but must NOT be identical — it is a different scheme
        @test isapprox(RP_explicit.plasma.ne, RP_implicit_1.plasma.ne, rtol = 5.0e-2)
        @test !isapprox(RP_explicit.plasma.ne, RP_implicit_1.plasma.ne, rtol = 1.0e-12)
    end
end

@testitem "Pure Diffusion: measured D matches configured D" setup = [PhysicsFixtures] begin
    # A Gaussian blob spreads by diffusion alone. Convection and sources are off, so the
    # growth of the blob's variance gives back the configured diffusivity:
    #     σ²(t) - σ²(0) = 2·D·t
    # Checked separately for pure perpendicular and pure parallel diffusion.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 50, NZ = 70,
        R_min = 0.8, R_max = 2.2,
        Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 100.0e-6,
        R0B0 = 1.0,
        Dpara0 = 0.0, Dperp0 = 100.0,          # overridden per block below
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0],
        wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        diffu = true,                     # ← the only transport term under test
        src = false,
        convec = false,
        ud_evolve = false,
        ud_method = "Xsec",
        Te_evolve = false,
        Ti_evolve = false,
        Ampere = false,
        E_para_self_ES = false,
        E_para_self_EM = false,
        Gas_evolve = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Coulomb_Collision = false,
        negative_n_correction = false,
        # The premise of this testitem is a SPATIALLY UNIFORM D: it weights the model's
        # tensor by the initial density and compares that against the realized
        # spreading, which is only the same number when D does not vary across the
        # blob. The flux ceiling makes it vary by construction — unlimited at the peak
        # where ∇n = 0, tight on the flanks — so the two legitimately diverge. Measured
        # here: model 43.3 vs realized 7.70 with the ceiling on, 56.6 vs 54.9 with it
        # off.
        #
        # That is not the limiter misbehaving. The assertion used to hold only because
        # `Dpara0` was added AFTER the cap and escaped it entirely; with `Dpara0 = 1e6`
        # against a ceiling of ¼v̄Lₙ ≈ 2.7e3 at one σ, closing that hole is the point of
        # the change. Switching the ceiling off here keeps this test measuring the
        # diffusion operator, which is what it is named for.
        limit_flux = (state = false, factor = 0.25),
    )

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2

    initialize!(RP)
    ini_ne = gaussian_density(RP.G; R0, Z0, σR = 0.1, σZ = 0.1, peak = 1.0e6)
    ini_ne[RP.G.nodes.on_out_wall_nids] .= 0.0

    function reset_to_initial_conditions!(RP, BR_ext, BZ_ext)
        RP.plasma.ne = copy(ini_ne)
        RP.fields.BR_ext .= BR_ext
        RP.fields.BZ_ext .= BZ_ext
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    measure_σ(ne) = (
        σR = sqrt(sum(ne .* (RP.G.R2D .- R0) .^ 2) / sum(ne)),
        σZ = sqrt(sum(ne .* (RP.G.Z2D .- Z0) .^ 2) / sum(ne)),
    )

    # Perpendicular only: with B purely toroidal the blob spreads isotropically in (R,Z)
    for implicit in (false, true)
        @testset "perpendicular only, Implicit=$implicit" begin
            RP.flags.Implicit = implicit
            RP.config.Dpara0 = 0
            RP.config.Dperp0 = 100

            initialize!(RP)
            reset_to_initial_conditions!(RP, 0.0, 0.0)
            RAPID2D.run_simulation!(RP)

            σR0, σZ0 = measure_σ(ini_ne)
            σR_end, σZ_end = measure_σ(RP.plasma.ne)
            mean_σ0 = (σR0 + σZ0) / 2
            mean_σ_end = (σR_end + σZ_end) / 2

            estimated_Dperp0 = (mean_σ_end^2 - mean_σ0^2) / (2.0 * RP.time_s)
            @test isapprox(estimated_Dperp0, RP.transport.Dperp0; rtol = 0.05)
        end
    end

    # Parallel only: a tilted B makes the spread anisotropic, so DRR and DZZ are
    # checked independently against the density-weighted diffusivity tensor.
    for implicit in (false, true)
        @testset "parallel only, Implicit=$implicit" begin
            RP.flags.Implicit = implicit
            RP.config.Dpara0 = 1.0e6
            RP.config.Dperp0 = 0

            initialize!(RP)
            reset_to_initial_conditions!(RP, 50.0e-4, 100.0e-4)
            RAPID2D.run_simulation!(RP)

            σR0, σZ0 = measure_σ(ini_ne)
            σR_end, σZ_end = measure_σ(RP.plasma.ne)

            avg_DRR = sum(RP.transport.DRR .* ini_ne) / sum(ini_ne)
            avg_DZZ = sum(RP.transport.DZZ .* ini_ne) / sum(ini_ne)
            estimated_DRR = (σR_end^2 - σR0^2) / (2.0 * RP.time_s)
            estimated_DZZ = (σZ_end^2 - σZ0^2) / (2.0 * RP.time_s)

            @test isapprox(avg_DRR, estimated_DRR; rtol = 0.05)
            @test isapprox(avg_DZZ, estimated_DZZ; rtol = 0.05)
        end
    end
end

@testitem "Free Accel & Heating: no collision" setup = [PhysicsFixtures] begin
    # Collisionless free acceleration in a static parallel E field. Transport, sources
    # and collisions are all off, so both species must reach exactly the ballistic
    # velocity  u = (q E_∥ / m)·t_end  — a direct check of the momentum equation.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 50, NZ = 70,
        R_min = 0.8, R_max = 2.2,
        Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 100.0e-6,
        R0B0 = 1.0,
        Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0],
        wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_evolve = true,                 # ← the momentum equation is under test
        ud_method = "Xsec",
        # no drag, no transport, no heating: acceleration must be purely ballistic
        Te_evolve = false, Ti_evolve = false,
        src = false, diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2

    initialize!(RP)
    ini_n = gaussian_density(RP.G; R0, Z0, σR = 0.1, σZ = 0.1, peak = 1.0e6)
    ini_n[RP.G.nodes.on_out_wall_nids] .= 0.0

    function reset_to_initial_conditions!(RP)
        RP.plasma.ne = copy(ini_n)
        RP.plasma.ni = copy(ini_n)
        RP.fields.BR_ext .= 1.0e-4
        RP.fields.BZ_ext .= 1.0e-4
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    for implicit in (false, true)
        @testset "Implicit=$implicit" begin
            RP.flags.Implicit = implicit
            RP.flags.Atomic_Collision = false     # collisionless: no momentum sink
            RP.flags.Include_ud_diffu_term = false
            initialize!(RP)
            reset_to_initial_conditions!(RP)

            RAPID2D.run_simulation!(RP)

            cnst = RP.config.constants
            ee, me, mi = cnst.ee, cnst.me, cnst.mi

            elec_accel_2D = @. -ee * RP.fields.E_para_ext / me
            expected_avg_ue_para = sum(@. ini_n * elec_accel_2D) / sum(ini_n) * RP.config.t_end_s

            ion_accel_2D = @. ee * RP.fields.E_para_ext / mi
            expected_avg_ui_para = sum(@. ini_n * ion_accel_2D) / sum(ini_n) * RP.config.t_end_s

            actual_avg_ue_para = sum(RP.plasma.ne .* RP.plasma.ue_para) / sum(RP.plasma.ne)
            @test isapprox(actual_avg_ue_para, expected_avg_ue_para; rtol = 0.01)
            actual_avg_ui_para = sum(RP.plasma.ni .* RP.plasma.ui_para) / sum(RP.plasma.ni)
            @test isapprox(actual_avg_ui_para, expected_avg_ui_para; rtol = 0.01)
        end
    end
end

@testitem "Ionization without transport" setup = [PhysicsFixtures] begin
    # Ionization in isolation: no transport, no heating, no field evolution, so density
    # changes come only from the source term. Two regimes are checked — below the
    # ionization threshold nothing happens at all, and above it the explicit and
    # implicit(θ=0) schemes must agree exactly.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 50, NZ = 70,
        R_min = 0.8, R_max = 2.2,
        Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 100.0e-6,
        R0B0 = 1.0,
        Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0],
        wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        src = true,                       # ← the ionization source is under test
        # no transport and no temperature evolution: Te is pinned by hand below
        ud_evolve = false, ud_method = "Xsec",
        Te_evolve = false, Ti_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = true
    RP.flags.Include_ud_diffu_term = false
    RP.flags.Ionz_method = "Xsec"

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2

    initialize!(RP)
    ini_n = gaussian_density(RP.G; R0, Z0, σR = 0.1, σZ = 0.1, peak = 1.0e6)
    ini_n[RP.G.nodes.on_out_wall_nids] .= 0.0

    function run_case!(RP; implicit, Te_eV, implicit_weight = nothing)
        RP.flags.Implicit = implicit
        if implicit_weight !== nothing
            RP.flags.θ_imp.transport = implicit_weight
            RP.flags.θ_imp.growth = implicit_weight
        end
        initialize!(RP)
        RP.plasma.ne = copy(ini_n)
        RP.plasma.ni = copy(ini_n)
        RP.fields.BR_ext .= 1.0e-4
        RP.fields.BZ_ext .= 1.0e-4
        RAPID2D.combine_external_and_self_fields!(RP)
        RP.plasma.Te_eV .= Te_eV
        RAPID2D.run_simulation!(RP)
        return RP.plasma
    end

    # Te = 0.1 eV is far below the 15.46 eV ionization threshold → nothing happens
    for implicit in (false, true)
        @testset "Te=0.1 eV gives no ionization, Implicit=$implicit" begin
            run_case!(RP; implicit, Te_eV = 0.1)
            @test all(RP.plasma.ν_en_iz .== 0.0)
            @test all(RP.operators.neRHS_src .== 0.0)
            @test ini_n == RP.plasma.ne
        end
    end

    @testset "Te=10 eV: implicit θ=0 ≡ explicit, θ=1 within 1%" begin
        explicit_plasma = deepcopy(run_case!(RP; implicit = false, Te_eV = 10.0))
        implicit_plasma_zeroθ = deepcopy(run_case!(RP; implicit = true, Te_eV = 10.0, implicit_weight = 0.0))
        implicit_plasma_oneθ = deepcopy(run_case!(RP; implicit = true, Te_eV = 10.0, implicit_weight = 1.0))

        @test isequal(explicit_plasma.ne, implicit_plasma_zeroθ.ne)
        # θ=1 differs by an O(ν_iz·dt) first-order scheme error (ν_iz,max·dt ≈ 7.5e-3
        # → ~0.6%). Ordinary numerics, not a bug.
        @test isapprox(explicit_plasma.ne, implicit_plasma_oneθ.ne, rtol = 1.0e-2)
    end
end

# SEQUENTIAL — do not split. Block (c) runs 100 MORE timesteps from the state block (b)
# left behind (t: 1 ms → 2 ms) and asserts the density has saturated RELATIVE to the
# 1 ms value. The later blocks are meaningless standalone.
@testitem "Thermal ionization at low/zero E/p (ClampExtrap low-field limit)" setup = [PhysicsFixtures] begin
    # E/p = 0 does NOT mean zero rate: ionization is set by the electron energy
    # distribution, so a 10 eV Maxwellian ionizes with no applied field at all.
    # ClampExtrap gives sub-minimum-E/p cells the low-field boundary rate; the old
    # fill-0 behaviour unphysically zeroed them. Te_evolve is ON here because the
    # energy cost of ionization cooling the electrons is part of what is being checked.
    #
    # Measured (implicit θ=1, dt=10µs): ne/n0 1.000 → 1.162 (saturated by 1 ms);
    # <Te> 10 → 2.00 (1 ms) → 1.81 eV (2 ms). Te cannot reach room_T here: below the
    # ~12 eV excitation threshold only elastic transfer (~2me/mi) remains. Thresholds
    # are loose so an RRC-table refresh does not break them.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-5, t_end_s = 2000.0e-6, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        src = true,                       # ← thermal ionization under test
        Te_evolve = true,                 # ← the cooling it causes is the other half
        ud_evolve = false, ud_method = "Xsec",
        Ti_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = true
    RP.flags.Ionz_method = "Xsec"
    RP.flags.Implicit = true
    RP.flags.θ_imp.transport = 1.0

    initialize!(RP)
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    ini_n = gaussian_density(RP.G; R0, Z0, σR = 0.1, σZ = 0.1, peak = 1.0e6)
    ini_n[RP.G.nodes.on_out_wall_nids] .= 0.0

    RP.plasma.ne = copy(ini_n)
    RP.plasma.ni = copy(ini_n)
    RP.fields.BR_ext .= 1.0e-4
    RP.fields.BZ_ext .= 1.0e-4
    RAPID2D.combine_external_and_self_fields!(RP)
    Te0 = 10.0
    RP.plasma.Te_eV .= Te0
    ini_sum = sum(RP.plasma.ne)

    @testset "(a) table clamps to the low-field boundary, threshold still applies" begin
        rrc_iz = RP.eRRCs.Ionization
        @test rrc_iz.itp(0.0, 15.0) == rrc_iz.itp(rrc_iz.EoverP[1], 15.0)  # clamped
        @test rrc_iz.itp(0.0, 15.0) > 0.0                                  # ...and nonzero
        @test rrc_iz.itp(0.0, 1.5) == 0.0    # 15.46 eV energy threshold still enforced
        update_transport_quantities!(RP)
        @test all(RP.plasma.ν_en_iz[RP.G.nodes.in_wall_nids] .> 0.0)
    end

    # (b) and (c) below CHAIN: (c) continues from (b)'s end state.
    #
    # `advance_timestep!` alone is NOT a complete step. The RRC-derived collision
    # frequencies are materialized once per step by `update_RRCs!`, which runs from
    # `update_transport_quantities!` — and `run_simulation!` calls that *after*
    # `advance_timestep!`, not inside it. Drop it and ν_en_iz stays pinned at the initial
    # Te = 10 eV, so the thermal ionization this test is about never shuts off.
    for _ in 1:100
        RAPID2D.advance_timestep!(RP, config.dt)
        update_transport_quantities!(RP)
    end
    ne_1ms = sum(RP.plasma.ne)
    Te_1ms = sum(RP.plasma.ne .* RP.plasma.Te_eV) / sum(RP.plasma.ne)
    @test ne_1ms > 1.05 * ini_sum   # density grew (measured ≈ 1.162×)
    @test Te_1ms < 0.6 * Te0        # ionization cost cooled the electrons (≈ 2.0 eV)

    for _ in 1:100
        RAPID2D.advance_timestep!(RP, config.dt)
        update_transport_quantities!(RP)
    end
    ne_2ms = sum(RP.plasma.ne)
    Te_2ms = sum(RP.plasma.ne .* RP.plasma.Te_eV) / sum(RP.plasma.ne)
    @test isapprox(ne_2ms, ne_1ms; rtol = 1.0e-2)     # ionization has shut off; ne saturated
    @test Te_2ms < Te_1ms                         # still cooling (measured 1.81 < 2.00)
    @test Te_2ms > RP.config.constants.room_T_eV  # but not below the gas temperature
    @test !any(isnan, RP.plasma.ne)
    @test !any(isnan, RP.plasma.Te_eV)
end

@testitem "Te relaxes to room_T_eV over ~tau_E, from both directions" setup = [PhysicsFixtures] begin
    # Elastic electron-neutral collisions equilibrate Te with the gas on
    #     tau_E = 1/(2·(me/mi)·nu_en_mom),   nu_en_mom = n_gas·RRC_mom_tot
    # Measured: nu_en_mom ≈ 1.70e4 /s, 2me/mi = 5.44e-4 ⇒ tau_E ≈ 0.108 s.
    # Both starting temperatures are far below the ionization threshold, so ne is
    # untouched and only the energy exchange is exercised. dt=1 ms is safe: the
    # relaxation is a smooth implicit exponential with dt/tau_E ≈ 0.01.
    FT = Float64

    # Builds a fresh, fully independent scenario at the given starting Te.
    function build_relaxation_case(Te0)
        # The explicit snapshot intervals are REQUIRED, not tuning: `dt` here is 1 ms, so
        # the 20 µs / 100 µs defaults would ask to record more often than the solver
        # steps, which `validate_config!` rejects.
        config = SimulationConfig{FT}(
            NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-3, t_end_s = 0.5, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
            prefilled_gas_pressure = 5.0e-3, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = scratch_output_dir()

        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(
            Te_evolve = true,             # ← electron energy equation under test
            src = true, ud_evolve = false, ud_method = "Xsec",
            Ti_evolve = false,
            diffu = false, convec = false, Ampere = false,
            E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
            update_ni_independently = false, Include_ud_convec_term = false,
            Coulomb_Collision = false, negative_n_correction = false,
        )
        RP.flags.Atomic_Collision = true   # ← elastic e-n transfer is the mechanism
        RP.flags.Ionz_method = "Xsec"
        RP.flags.Implicit = true
        RP.flags.θ_imp.transport = 1.0

        initialize!(RP)
        R0 = (config.R_min + config.R_max) / 2
        Z0 = (config.Z_min + config.Z_max) / 2
        ini_n = gaussian_density(RP.G; R0, Z0, σR = 0.1, σZ = 0.1, peak = 1.0e6)
        ini_n[RP.G.nodes.on_out_wall_nids] .= 0.0
        RP.plasma.ne = copy(ini_n)
        RP.plasma.ni = copy(ini_n)
        RP.fields.BR_ext .= 1.0e-4
        RP.fields.BZ_ext .= 1.0e-4
        RAPID2D.combine_external_and_self_fields!(RP)
        RP.plasma.Te_eV .= Te0
        return RP, config, sum(ini_n)
    end

    weighted_Te(RP) = sum(RP.plasma.ne .* RP.plasma.Te_eV) / sum(RP.plasma.ne)

    @testset "tau_E from the momentum-transfer rate" begin
        RP0, _, _ = build_relaxation_case(0.026)
        update_transport_quantities!(RP0)
        me, mi = RP0.config.constants.me, RP0.config.constants.mi
        inw = RP0.G.nodes.in_wall_nids
        ν_mom = sum(RP0.plasma.ν_en_mom_tot[inw]) / length(inw)
        τ_E = 1 / (2 * (me / mi) * ν_mom)
        @test ν_mom > 0.0
        @test 0.01 < τ_E < 1.0        # measured ≈ 0.108 s
    end

    # 500 × 1 ms = 0.5 s ≈ 4.6 tau_E — long enough to converge from either side.
    nsteps = 500
    for (Te0, is_hot) in ((0.1, true), (0.001, false))
        @testset "Te0 = $Te0 eV ($(is_hot ? "cools" : "heats")) onto room_T_eV" begin
            RP, config, ini_sum = build_relaxation_case(Te0)
            room = RP.config.constants.room_T_eV
            # `update_transport_quantities!` is the other half of a step — it is what
            # re-materializes ν_en_mom_tot, and here that rate sets the relaxation time.
            for _ in 1:nsteps
                RAPID2D.advance_timestep!(RP, config.dt)
                update_transport_quantities!(RP)
            end
            Te_end = weighted_Te(RP)

            @test isapprox(Te_end, room; rtol = 0.1)   # within 1.3% (hot) / 3.9% (cold)
            if is_hot
                @test Te_end < Te0                    # hot electrons cooled by the gas
            else
                @test Te_end > Te0                    # cold electrons HEATED by the gas
            end

            # Far below the ionization threshold → density untouched
            @test isapprox(sum(RP.plasma.ne), ini_sum; rtol = 1.0e-6)
            @test all(RP.plasma.ν_en_iz .== 0.0)
            @test !any(isnan, RP.plasma.Te_eV)
        end
    end
end

@testitem "P_ela is charged with the ELASTIC share of the drift friction" setup = [PhysicsFixtures] begin
    # The 2me/mi recoil is handed to the molecule by ELASTIC momentum-transfer collisions
    # only; the inelastic share of Total_Momentum carries its momentum into excitation and
    # ionization, which the same function charges separately as P_exc / P_iz. Spending the
    # total on P_ela counts those twice.
    #
    # The relaxation scenario above cannot catch this: it runs at Te ~ 0.026-0.1 eV, far
    # below the ~9 eV excitation threshold, where no inelastic channel is open and the two
    # moments coincide. This item picks a state where they demonstrably do not.
    FT = Float64
    p_gas = 5.0e-3                       # Pa; E/p = E_para / p_gas
    config = SimulationConfig{FT}(
        NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 1.0e-6, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = p_gas,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        Te_evolve = false, src = false, ud_evolve = false, ud_method = "Xsec",
        Ti_evolve = false, diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = true
    initialize!(RP)

    # Erg = 1.5*Te (u_para = 0) ~ 14 eV, E/p = 0.5 / 5e-3 = 100 -> elastic share ~0.7
    RP.plasma.Te_eV .= 9.3
    RP.fields.E_para_tot .= 0.5
    # The heating powers read the materialized ν_en_*, so hand-setting the state is not
    # enough — update_RRCs! is what carries it onto the reaction-rate surfaces.
    update_RRCs!(RP)
    RAPID2D.update_electron_heating_powers!(RP)

    K_tot = get_electron_RRC(RP, :Total_Momentum)
    K_ela = get_electron_RRC(RP, :Momentum_by_ela)
    inw = RP.G.nodes.in_wall_nids

    # The chosen state must actually separate the two moments, or this test proves nothing.
    @test all(K_ela[inw] .< K_tot[inw])
    @test maximum(K_ela[inw] ./ K_tot[inw]) < 0.9

    (; ee, me, mi) = RP.config.constants
    pla = RP.plasma
    recoil = @. (2 * me / mi) * pla.n_H2_gas * 1.5 * (pla.Te_eV - pla.T_gas_eV) * ee
    @test pla.ePowers.ela[inw] ≈ (recoil .* K_ela)[inw]
    # ...and is NOT the total-momentum version, which is what the bug computed.
    @test !isapprox(pla.ePowers.ela[inw], (recoil .* K_tot)[inw]; rtol = 0.05)
end

@testitem "ud_method = Xsec_fit solves the instantaneous Drude balance" setup = [PhysicsFixtures] begin
    # The alternative drift method. Unlike "Xsec" it does not advance u_para; it solves
    # qE = m ν u algebraically, and it omits the ν_iz dilution term — which is exactly
    # what makes it useful for isolating that term. It had no test, so the whole branch
    # was dark. Closed form, so the assertions are exact rather than tolerance-based.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 1.0e-6, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_method = "Xsec_fit", ud_evolve = true,
        Te_evolve = false, Ti_evolve = false, src = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = true
    initialize!(RP)
    RP.plasma.Te_eV .= 9.3
    RP.fields.E_para_tot .= 0.5

    inw = RP.G.nodes.in_wall_nids
    drude(ν) = @. -RP.config.ee * RP.fields.E_para_tot / (RP.config.me * ν)

    # update_ue_para! reads the MATERIALIZED ν_en_mom_tot, so update_RRCs! is what moves
    # the lookup — a step is [update_RRCs! ... update_ue_para!], and inside that step the
    # frequency is frozen. Calling update_ue_para! twice without an update_RRCs! between
    # them would solve the same Drude balance twice, which is the point of freezing it.
    ν_en_mom() = copy(RP.plasma.ν_en_mom_tot)

    update_RRCs!(RP)
    ν1 = ν_en_mom()                       # queried at ue_para = 0
    RAPID2D.update_ue_para!(RP)
    @test RP.plasma.ue_para[inw] ≈ drude(ν1)[inw]
    u_neutral_only = copy(RP.plasma.ue_para)

    # Coulomb collisions add to the same denominator, Spitzer-weighted.
    RP.flags.Coulomb_Collision = true
    RP.flags.Spitzer_Resistivity = true
    RP.plasma.ν_ei .= 1.0e5
    RP.plasma.sptz_fac .= 0.5
    update_RRCs!(RP)                      # next step: Erg now carries the drift energy
    ν2 = ν_en_mom() .+ 0.5 .* 1.0e5
    RAPID2D.update_ue_para!(RP)
    @test RP.plasma.ue_para[inw] ≈ drude(ν2)[inw]
    @test all(abs.(RP.plasma.ue_para[inw]) .< abs.(u_neutral_only[inw]))   # more drag, less drift

    # Within a step the frequency does not respond to the drift it just produced: solving
    # again without a new update_RRCs! reproduces the same u exactly. This is the
    # single-evaluation-point invariant, stated where it is easiest to see.
    u_frozen = copy(RP.plasma.ue_para)
    RAPID2D.update_ue_para!(RP)
    @test RP.plasma.ue_para[inw] == u_frozen[inw]
end

# ── RRC evaluation point ─────────────────────────────────────────────────────────────

@testitem "update_RRCs! is the single RRC evaluation point of a step" setup = [PhysicsFixtures] begin
    # The invariant: every (E/p, Ebar) surface is read exactly once per step, by
    # update_RRCs!, and every consumer reads the materialized nu_en_* fields. Consumers
    # used to re-query partway through advance_timestep!, at (Te^n, u^{n+1}, E(t_n)) --
    # a state the plasma never occupies, because in Ebar = 1.5*Te + 0.5*me*u^2/e the
    # drift term had advanced and the thermal term had not.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 1.0e-6, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        Te_evolve = true, src = true, ud_evolve = true, ud_method = "Xsec",
        Ti_evolve = false, diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = true
    RP.flags.Ionz_method = "Xsec"
    initialize!(RP)

    # Erg ~ 14 eV, E/p = 0.5 / 5e-3 = 100: high enough that inelastic channels are open,
    # so the four surfaces are genuinely distinct.
    RP.plasma.Te_eV .= 9.3
    RP.fields.E_para_tot .= 0.5
    RP.plasma.ue_para .= -3.0e5

    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids
    ν_fields() = (
        copy(pla.ν_en_iz), copy(pla.ν_en_mom_tot),
        copy(pla.ν_en_mom_ela), copy(pla.ν_en_exc_eff),
    )

    @testset "all four surfaces are populated, and carry the n_gas factor" begin
        update_RRCs!(RP)
        for f in ν_fields()
            @test all(f[inw] .> 0.0)
        end
        @test pla.ν_en_mom_tot[inw] ≈
            (pla.n_H2_gas .* get_electron_RRC(RP, :Total_Momentum))[inw]
        # At E/p ~ 100 elastic is only ~70% of the drift friction, so the two momentum
        # surfaces must not coincide -- otherwise the rest of this item proves nothing.
        @test all(pla.ν_en_mom_ela[inw] .< pla.ν_en_mom_tot[inw])
        @test all(pla.ν_en_iz[RP.G.nodes.on_out_wall_nids] .== 0.0)
    end

    @testset "idempotent: a second call at the same state changes nothing" begin
        update_RRCs!(RP)
        before = ν_fields()
        update_RRCs!(RP)
        @test all(a == b for (a, b) in zip(before, ν_fields()))
    end

    @testset "the frequency does not follow the state within a step" begin
        update_transport_quantities!(RP)     # materializes nu AND derives ueR/uephi/ueZ
        ν_entry = copy(pla.ν_en_mom_tot)
        ue_mag_sq = @. pla.ueR^2 + pla.ueϕ^2 + pla.ueZ^2
        @test all(ue_mag_sq[inw] .> 0.0)     # or the drag assertion below is vacuous

        # advance_timestep! advances u_para (update_ue_para!) before the energy equation
        # runs, so mid-step the lookup coordinate has moved. Imitate that move directly.
        pla.ue_para .*= 5.0
        ν_mid = pla.n_H2_gas .* get_electron_RRC(RP, :Total_Momentum)
        @test !isapprox(ν_mid[inw], ν_entry[inw]; rtol = 1.0e-3)   # the state really moved
        @test pla.ν_en_mom_tot == ν_entry                          # ...the field did not

        # ePowers.drag adds a Spitzer term regardless of flags.Coulomb_Collision, and
        # initialize! leaves nu_ei/sptz_fac populated, so it contributes ~6e-8 of the
        # total here. Zero it to isolate the neutral drag this item is about.
        pla.ν_ei .= 0.0

        update_electron_heating_powers!(RP)
        (; me, ee, char_exc_erg_eV, iz_erg_eV) = RP.config.constants

        # P_drag is charged at the entry-state frequency -- the same number the momentum
        # equation used to remove that momentum.
        @test pla.ePowers.drag[inw] ≈ (me .* ue_mag_sq .* ν_entry)[inw]
        @test !isapprox(
            pla.ePowers.drag[inw], (me .* ue_mag_sq .* ν_mid)[inw]; rtol = 1.0e-3
        )

        # The inelastic channels read their own materialized frequencies. nu_en_exc_eff is
        # normalized to char_exc_erg_eV, which is why that constant appears bare here.
        @test pla.ePowers.exc[inw] ≈ (ee * char_exc_erg_eV .* pla.ν_en_exc_eff)[inw]
        @test pla.ePowers.iz[inw] ≈ (ee * iz_erg_eV .* pla.ν_en_iz)[inw]
    end

    @testset "advance_timestep! does not move the frequencies" begin
        update_transport_quantities!(RP)
        before = ν_fields()
        RAPID2D.advance_timestep!(RP, config.dt)
        @test all(a == b for (a, b) in zip(before, ν_fields()))

        # ...and the transport update that closes the step does move them, because Te and
        # u_para advanced. Without this the assertion above could pass on a frozen state.
        update_transport_quantities!(RP)
        @test any(a != b for (a, b) in zip(before, ν_fields()))
    end
end

@testitem "update_RRCs! covers Atomic_Collision and src independently" setup = [PhysicsFixtures] begin
    # nu_en_iz feeds two consumers behind two DIFFERENT flags: the parallel momentum drag
    # (Atomic_Collision) and the continuity source (src). Its only unconditional writer
    # used to sit under Atomic_Collision, so with just `src` set the field held whatever
    # the previous iteration's mid-step query had left there. Nothing covered that
    # combination.
    FT = Float64
    config = SimulationConfig{FT}(
        NR = 20, NZ = 30, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
        dt = 1.0e-6, t_end_s = 1.0e-6, R0B0 = 1.0, Dpara0 = 0.0, Dperp0 = 0.0,
        prefilled_gas_pressure = 5.0e-3,
        wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
    )
    config.Output_path = scratch_output_dir()

    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        Te_evolve = false, src = true, ud_evolve = false, ud_method = "Xsec",
        Ti_evolve = false, diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false, Gas_evolve = false,
        update_ni_independently = false, Include_ud_convec_term = false,
        Coulomb_Collision = false, negative_n_correction = false,
    )
    RP.flags.Atomic_Collision = false     # ← ionization on, neutral collisions off
    RP.flags.Ionz_method = "Xsec"
    initialize!(RP)
    @test !RP.flags.Atomic_Collision      # gas is present, so initialize! left it off

    pla = RP.plasma
    inw = RP.G.nodes.in_wall_nids
    pla.Te_eV .= 9.3
    update_RRCs!(RP)

    @test all(pla.ν_en_iz[inw] .> 0.0)                            # still evaluated
    @test all(pla.ν_en_iz[RP.G.nodes.on_out_wall_nids] .== 0.0)
    @test all(pla.ν_en_mom_tot .== 0.0)                           # neutral drag stays off
    @test all(pla.ν_en_mom_ela .== 0.0)
    @test all(pla.ν_en_exc_eff .== 0.0)

    # ...and it tracks the state instead of going stale.
    ν1 = copy(pla.ν_en_iz)
    pla.Te_eV .= 20.0
    update_RRCs!(RP)
    @test all(pla.ν_en_iz[inw] .> ν1[inw])
end

# ── Ion energetics ───────────────────────────────────────────────────────────────────

# SEQUENTIAL — do not split. The trailing blocks all `deepcopy(RP)` and therefore INHERIT
# the state left by the last nested @testset ("ui=0, Ti>T_gas": T_gas_eV=1.0, Ti_eV=10.0).
# In particular the RP_no_src assertion `iPowers.atomic .!= 0.0` requires Ti ≠ T_gas; on a
# freshly built RP both equal room_T_eV, atomic power is identically 0, and it FAILS.
@testitem "Ion Heating Powers" begin
    using RAPID2D.Statistics

    # Sign conventions of the two ion power channels, checked against hand-reasoned
    # limits: `atomic` (elastic + charge exchange with the neutral gas) and `equi`
    # (Coulomb equilibration with the electrons).
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 20, NZ = 20,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 1.0e-6,
        Dperp0 = 001,
    )
    RP = RAPID{Float64}(config)
    RP.flags.src = true
    RP.flags.Coulomb_Collision = true    # ← enables the `equi` channel
    RP.flags.Atomic_Collision = true     # ← enables the `atomic` channel
    RP.flags.Ti_evolve = true
    initialize!(RP)

    # High density so both channels are numerically visible
    RP.plasma.ne .= 1.0e18
    RP.plasma.ni .= RP.plasma.ne

    room_T_eV = RP.config.constants.room_T_eV

    @test size(RP.plasma.iPowers.tot) == (RP.G.NR, RP.G.NZ)
    @test size(RP.plasma.iPowers.atomic) == (RP.G.NR, RP.G.NZ)
    @test size(RP.plasma.iPowers.equi) == (RP.G.NR, RP.G.NZ)

    @testset "ui=0, Ti=T_gas, Te>Ti" begin
        RP.plasma.ui_para .= 0.0
        RP.plasma.Ti_eV .= room_T_eV
        RP.plasma.T_gas_eV = room_T_eV
        RP.plasma.Te_eV .= 10.0

        update_transport_quantities!(RP)
        update_coulomb_collision_parameters!(RP)
        update_ion_heating_powers!(RP)

        # ui=0 and Ti=T_gas ⇒ no atomic exchange at all
        @test mean(RP.plasma.iPowers.atomic) == 0.0
        # Ti < Te ⇒ electrons heat the ions
        @test mean(RP.plasma.iPowers.equi) > 0.0
    end
    @testset "ui=1e3, Ti=T_gas, Ti<Te" begin
        RP.plasma.ui_para .= 1.0e3
        RP.plasma.Ti_eV .= 1.0
        RP.plasma.T_gas_eV = 1.0
        RP.plasma.Te_eV .= 0.1

        update_transport_quantities!(RP)
        update_coulomb_collision_parameters!(RP)
        update_ion_heating_powers!(RP)

        in_wall_nids = RP.G.nodes.in_wall_nids
        # Ion flow through the neutral gas heats the ions
        @test all(RP.plasma.iPowers.atomic[in_wall_nids] .> 0.0)
        # Ti > Te ⇒ ions lose energy to the electrons
        @test all(RP.plasma.iPowers.equi[in_wall_nids] .< 0.0)
    end
    @testset "ui=0, Ti>T_gas, Ti=Te" begin
        RP.plasma.ui_para .= 0.0
        RP.plasma.Ti_eV .= 10.0
        RP.plasma.T_gas_eV = 1.0
        RP.plasma.Te_eV .= 10.0

        update_transport_quantities!(RP)
        update_coulomb_collision_parameters!(RP)
        update_ion_heating_powers!(RP)

        in_wall_nids = RP.G.nodes.in_wall_nids
        # Ti > T_gas with no flow ⇒ ions cool against the gas
        @test all(RP.plasma.iPowers.atomic[in_wall_nids] .< 0.0)
        # Ti = Te ⇒ no net Coulomb exchange
        @test all(RP.plasma.iPowers.equi[in_wall_nids] .== 0.0)
    end

    # Powers must vanish outside the wall
    out_wall_idx = RP.G.nodes.out_wall_nids
    if !isempty(out_wall_idx)
        @test all(RP.plasma.iPowers.tot[out_wall_idx] .== 0.0)
        @test all(RP.plasma.iPowers.atomic[out_wall_idx] .== 0.0)
        @test all(RP.plasma.iPowers.equi[out_wall_idx] .== 0.0)
    end

    # Disabling the source removes the ionization contribution but leaves elastic and
    # charge exchange. NOTE: relies on Ti_eV=10.0 / T_gas_eV=1.0 left by the testset
    # above — see the SEQUENTIAL warning on this testitem.
    RP_no_src = deepcopy(RP)
    RP_no_src.flags.src = false
    update_ion_heating_powers!(RP_no_src)
    in_wall_nids = RP_no_src.G.nodes.in_wall_nids
    @test all(RP_no_src.plasma.iPowers.atomic[in_wall_nids] .!= 0.0)

    # Disabling Coulomb collisions removes the equilibration channel entirely
    RP_no_coulomb = deepcopy(RP)
    RP_no_coulomb.flags.Coulomb_Collision = false
    update_ion_heating_powers!(RP_no_coulomb)
    @test all(RP_no_coulomb.plasma.iPowers.equi .== 0.0)

    # Hotter ions lose more energy to the gas than colder ones
    RP_hot = deepcopy(RP)
    RP_hot.plasma.Ti_eV .= 15.0
    update_ion_heating_powers!(RP_hot)
    RP_cold = deepcopy(RP)
    RP_cold.plasma.Ti_eV .= 0.01
    update_ion_heating_powers!(RP_cold)
    in_wall_nids = RP_hot.G.nodes.in_wall_nids
    @test mean(RP_hot.plasma.iPowers.atomic[in_wall_nids]) < mean(RP_cold.plasma.iPowers.atomic[in_wall_nids])
end

# SEQUENTIAL — do not split. Four `run_simulation!` calls chain with NO time reset:
# run_simulation! loops `while RP.time_s < t_end`, so each call RESUMES from the previous
# end state as t_end_s is raised 50e-6 → 1e-3 → 5e-3 → 40e-3 (with RP.dt *= 10 before the
# last). The absolute goldens are only meaningful on the accumulated trajectory.
@testitem "Te-Ti equilibration by Coulomb_Collision" begin
    using RAPID2D.Statistics
    using RAPID2D.SimpleUnPack

    # Pure Coulomb equilibration between a 1 eV electron population and cold ions.
    # Atomic collisions and all transport are off, so the temperature difference must
    # decay as the analytic  ΔT(t) = ΔT₀·exp(-2t/τ_eq)  and end at the mean of the two.
    FT = Float64
    config = SimulationConfig{FT}(
        device_Name = "manual",
        NR = 20, NZ = 20,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 10.0e-6,
        t_end_s = 10.0e-3,
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        Coulomb_Collision = true,         # ← the only energy exchange channel
        Te_evolve = true, Ti_evolve = true,
        Atomic_Collision = false,         # no gas coupling: the two species only see each other
        src = false, convec = false, diffu = false, ud_evolve = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false,
        Include_Te_convec_term = false, update_ni_independently = false,
        Gas_evolve = false, Ampere = false, E_para_self_ES = false,
    )
    initialize!(RP)

    # ne chosen so that ν_ei ≈ 1e5 /s, i.e. τ_ei = 10 µs (asserted below)
    RP.plasma.ne .= 2.841e15
    RP.plasma.ni .= RP.plasma.ne
    RP.plasma.Te_eV .= 1.0
    RP.plasma.Ti_eV .= 1.0e-6

    update_coulomb_collision_parameters!(RP)

    in_wall_nids = RP.G.nodes.in_wall_nids
    @unpack mi, me = RP.config.constants

    avg_ini_Ti = mean(RP.plasma.Ti_eV[in_wall_nids])
    avg_ini_Te = mean(RP.plasma.Te_eV[in_wall_nids])
    avg_ini_τ_ei = 1.0 ./ mean(RP.plasma.ν_ei)
    avg_ini_τ_eq = 0.5 * ((mi + me)^2 / (mi * me)) * avg_ini_τ_ei

    @test isapprox(mean(RP.plasma.ν_ei), 1.0e5, rtol = 1.0e-4)
    @test isapprox(avg_ini_τ_ei, 10.0e-6, rtol = 1.0e-4)

    ΔT0 = abs(avg_ini_Te - avg_ini_Ti)
    analytic_ΔT = (τeq, t) -> ΔT0 * exp(-2 * t / τeq)
    measure_ΔT = () -> mean(RP.plasma.Te_eV[in_wall_nids]) - mean(RP.plasma.Ti_eV[in_wall_nids])

    # Each run_simulation! RESUMES from the previous end state (no time reset).
    RP.t_end_s = 50.0e-6
    run_simulation!(RP)
    @test isapprox(analytic_ΔT(avg_ini_τ_eq, RP.time_s), measure_ΔT(), rtol = 1.0e-3)

    RP.t_end_s = 1.0e-3
    run_simulation!(RP)
    @test isapprox(analytic_ΔT(avg_ini_τ_eq, RP.time_s), measure_ΔT(), rtol = 1.0e-2)

    RP.t_end_s = 5.0e-3
    run_simulation!(RP)
    @test isapprox(mean(RP.plasma.Te_eV[in_wall_nids]), 0.7581, atol = 0.01)
    @test isapprox(mean(RP.plasma.Ti_eV[in_wall_nids]), 0.2425, atol = 0.01)

    # Much longer, with a coarser timestep: both must settle at the mean, 0.5 eV
    RP.dt *= 10
    RP.t_end_s = 40.0e-3
    run_simulation!(RP)
    @test isapprox(mean(RP.plasma.Te_eV[in_wall_nids]), 0.5, atol = 0.01)
    @test isapprox(mean(RP.plasma.Ti_eV[in_wall_nids]), 0.5, atol = 0.01)

    @test isapprox(RP.plasma.Te_eV[in_wall_nids], RP.plasma.Ti_eV[in_wall_nids], rtol = 1.0e-3)
end

# ── Parallel momentum: Coulomb drag in the time-evolving ("Xsec") method ─────────────
#
# `ud_method` picks HOW the parallel drift is obtained, not WHICH physics acts on it:
#   "Lloyd_fit" — empirical  u∥ = 5719·(−E∥/p)
#   "Xsec_fit"  — instantaneous force balance  u∥ = qe·E∥/(me·ν_tot)   (algebraic)
#   "Xsec"      — the momentum ODE integrated in time                  (default)
# The three must therefore agree on the STEADY STATE. The items below pin that
# equivalence for "Xsec", which is the only method whose drag enters as a coefficient
# rather than a denominator — and therefore the only one where a missing collision
# channel fails silently instead of changing an obvious division.

@testitem "Xsec momentum relaxes to the parallel force balance (Coulomb ON)" setup = [PhysicsFixtures] begin
    # Governing equation (Yoo, IFPC 2024):
    #   du∥/dt = qe·E∥/me − (ν_mom + ν_iz)·u∥ − ξ_sptz·ν_ei·(u∥ − u_i∥)
    # Steady state with the ions held at rest is therefore exactly
    #   u∥ = qe·E∥ / [ me·(ν_mom + ν_iz + ν_ei_eff) ]        , ν_ei_eff ≡ ξ_sptz·ν_ei
    # i.e. the same balance "Xsec_fit" solves algebraically.
    #
    # The collision frequencies are frozen (update_transport_quantities! is called ONCE,
    # never inside the loop), the ions are pinned at rest and every spatial term is off,
    # so the relaxation is exactly linear and the drag is the only physics under test.
    FT = Float64
    config = SimulationConfig{FT}(
        device_Name = "manual",
        NR = 12, NZ = 12,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 2.0e-6,
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_evolve = true, ud_method = "Xsec", Implicit = true,
        Atomic_Collision = true,          # ← ν_en_mom_tot: the neutral drag
        Coulomb_Collision = true,         # ← ν_ei: the channel under test
        Spitzer_Resistivity = true,       # ← ξ_sptz weighting
        src = false,                      # no ionization: ν_iz = 0, ne stays put
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Include_ud_pressure_term = false,
        Include_ud_diffu_term = false,
    )
    initialize!(RP)

    RP.plasma.ne .= 1.0e18
    RP.plasma.ni .= 1.0e18
    RP.plasma.Te_eV .= 10.0
    RP.plasma.Ti_eV .= 1.0
    RP.fields.E_para_tot .= 1.0                  # uniform parallel field [V/m]

    RAPID2D.update_transport_quantities!(RP)     # populate the ν's once, then freeze
    ν_mom = copy(RP.plasma.ν_en_mom_tot)
    ν_iz = copy(RP.plasma.ν_en_iz)
    ν_ei_eff = copy(RP.plasma.ν_ei_eff)

    inw = RP.G.nodes.in_wall_nids
    # Regime check: this is the Coulomb-dominated corner, so a missing ν_ei cannot hide
    # inside the neutral drag.
    @test all(ν_ei_eff[inw] .> 0)
    @test minimum(ν_ei_eff[inw]) > maximum(ν_mom[inw])

    RP.plasma.ue_para .= 0.0
    RP.plasma.ui_para .= 0.0
    for _ in 1:400
        RAPID2D.update_ue_para!(RP)
        RP.plasma.ui_para .= 0.0                 # ions pinned: isolate the electron drag
    end

    qe, me = RP.config.constants.qe, RP.config.constants.me
    expected = @. qe * RP.fields.E_para_tot / (me * (ν_mom + ν_iz + ν_ei_eff))
    @test isapprox(RP.plasma.ue_para[inw], expected[inw]; rtol = 1.0e-6)
end

@testitem "Xsec momentum relaxes to the neutral-only balance (Coulomb OFF)" setup = [PhysicsFixtures] begin
    # Same scenario with the Coulomb channel switched off: the steady state must drop
    # the ν_ei term and land on the neutral-only balance, and must be strictly FASTER
    # (less drag ⇒ more drift) than the Coulomb-on case above.
    FT = Float64
    config = SimulationConfig{FT}(
        device_Name = "manual",
        NR = 12, NZ = 12,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 2.0e-6,
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_evolve = true, ud_method = "Xsec", Implicit = true,
        Atomic_Collision = true,
        Coulomb_Collision = false,        # ← the only difference
        src = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Include_ud_pressure_term = false,
        Include_ud_diffu_term = false,
    )
    initialize!(RP)

    RP.plasma.ne .= 1.0e18
    RP.plasma.ni .= 1.0e18
    RP.plasma.Te_eV .= 10.0
    RP.plasma.Ti_eV .= 1.0
    RP.fields.E_para_tot .= 1.0

    RAPID2D.update_transport_quantities!(RP)
    ν_mom = copy(RP.plasma.ν_en_mom_tot)
    ν_iz = copy(RP.plasma.ν_en_iz)

    RP.plasma.ue_para .= 0.0
    RP.plasma.ui_para .= 0.0
    for _ in 1:400
        RAPID2D.update_ue_para!(RP)
        RP.plasma.ui_para .= 0.0
    end

    qe, me = RP.config.constants.qe, RP.config.constants.me
    expected = @. qe * RP.fields.E_para_tot / (me * (ν_mom + ν_iz))
    inw = RP.G.nodes.in_wall_nids
    @test isapprox(RP.plasma.ue_para[inw], expected[inw]; rtol = 1.0e-6)
end

@testitem "Parallel momentum decays at 1/(nu_mom + nu_iz + nu_ei)" setup = [PhysicsFixtures] begin
    # Same shape as the "Te relaxes over ~tau_E" item, for MOMENTUM: switch the drive
    # off (E∥ = 0), give the electrons an initial drift and watch it decay. With the
    # ions pinned at rest the momentum equation is exactly
    #     du∥/dt = -(nu_mom + nu_iz + nu_ei_eff)·u∥  ==>  u∥(t) = u0·exp(-t/tau)
    # so the decay time DIRECTLY measures the drag coefficient: a missing collision
    # channel shows up as a decay that is too slow, at every sampled time.
    #
    # dt = tau/100 keeps the backward-Euler factor (1+dt/tau)^-n within ~1% of exp(-t/tau)
    # out to 3 tau, so the +/-10% windows below are comfortably scheme-independent.
    FT = Float64
    config = SimulationConfig{FT}(
        device_Name = "manual",
        NR = 12, NZ = 12,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 1.0e-9,                      # overwritten below once tau is known
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_evolve = true, ud_method = "Xsec", Implicit = true,
        Atomic_Collision = true,
        Coulomb_Collision = true, Spitzer_Resistivity = true,
        src = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Include_ud_pressure_term = false,
        Include_ud_diffu_term = false,
    )
    initialize!(RP)

    RP.plasma.ne .= 1.0e18
    RP.plasma.ni .= 1.0e18
    RP.plasma.Te_eV .= 10.0
    RP.plasma.Ti_eV .= 1.0
    RP.fields.E_para_tot .= 1.0          # only to give update_RRCs! a sane E/p...
    RAPID2D.update_transport_quantities!(RP)
    RP.fields.E_para_tot .= 0.0          # ...then switch the drive OFF: pure decay

    inw = RP.G.nodes.in_wall_nids
    ν_tot = @. RP.plasma.ν_en_mom_tot + RP.plasma.ν_en_iz + RP.plasma.ν_ei_eff
    τ = 1 / (sum(ν_tot[inw]) / length(inw))
    @test minimum(RP.plasma.ν_ei_eff[inw]) > maximum(RP.plasma.ν_en_mom_tot[inw])
    @test 1.0e-9 < τ < 1.0e-3            # sanity band for this regime

    u0 = 1.0e5
    RP.dt = τ / 100
    RP.plasma.ue_para .= u0
    RP.plasma.ui_para .= 0.0

    # (t/tau, lower, upper) — +/-10% windows around exp(-t/tau)
    checkpoints = [(0.5, 0.9, 1.1), (1.0, 0.9, 1.1), (2.0, 0.9, 1.1), (3.0, 0.9, 1.1)]
    targets = [round(Int, n_tau * 100) for (n_tau, _, _) in checkpoints]
    steps_each = [targets[1]; diff(targets)]      # no cross-iteration mutable state
    for (k, (n_tau, lo, hi)) in enumerate(checkpoints)
        for _ in 1:steps_each[k]
            RAPID2D.update_ue_para!(RP)
            RP.plasma.ui_para .= 0.0
        end
        ratio = sum(RP.plasma.ue_para[inw]) / length(inw) / u0
        @test lo * exp(-n_tau) < ratio < hi * exp(-n_tau)
    end
    # and it really is a decay, not a sign flip or a stall
    @test 0.0 < sum(RP.plasma.ue_para[inw]) / length(inw) < 0.1 * u0
end

@testitem "e-i Coulomb friction conserves momentum and equalises u_e, u_i" setup = [PhysicsFixtures] begin
    # The most fundamental statement about the Coulomb term: it is an INTERNAL exchange.
    # With E∥ = 0 and no neutral drag it is the only channel acting, so
    #   d/dt [ me·ne·u_e + mi·ni·u_i ] = 0        (exactly, when ne = ni)
    # and both species must relax onto the common centre-of-mass velocity
    #   u_cm = (me·u_e0 + mi·u_i0)/(me + mi)
    # on the timescale 1/nu_ei_eff.
    #
    # This is the check that a one-sided friction cannot survive: dropping the -nu_ei·u_e
    # sink from the electron equation leaves du_e/dt = +nu_ei·u_i, which (i) never lets
    # u_e decay and (ii) CREATES momentum at the rate me·n·nu_ei·u_e out of nothing.
    #
    # NOTE this is a zero-current relaxation. With E∥ != 0 the two velocities must NOT
    # converge — the steady relative drift IS the plasma current, set by Spitzer
    # resistivity, which the force-balance items above pin down.
    FT = Float64
    config = SimulationConfig{FT}(
        device_Name = "manual",
        NR = 12, NZ = 12,
        prefilled_gas_pressure = 5.0e-3,
        R0B0 = 1.0,
        dt = 1.0e-9,                      # rewritten below once tau is known
    )
    RP = RAPID{FT}(config)
    RP.flags = SimulationFlags{FT}(
        ud_evolve = true, ud_method = "Xsec", Implicit = true,
        Atomic_Collision = false,         # ← no neutral drag: Coulomb is the ONLY channel
        Coulomb_Collision = true, Spitzer_Resistivity = true,
        src = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
        diffu = false, convec = false, Ampere = false,
        E_para_self_ES = false, E_para_self_EM = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Include_ud_pressure_term = false,
        Include_ud_diffu_term = false,
    )
    initialize!(RP)

    n0 = 1.0e18
    RP.plasma.ne .= n0
    RP.plasma.ni .= n0                    # ne == ni is what makes the exchange exact
    RP.plasma.Te_eV .= 10.0
    RP.plasma.Ti_eV .= 1.0
    RAPID2D.update_transport_quantities!(RP)
    RP.fields.E_para_tot .= 0.0           # no drive: pure internal relaxation

    inw = RP.G.nodes.in_wall_nids
    @test all(RP.plasma.ν_en_mom_tot[inw] .== 0.0)   # neutral drag really is off
    @test all(RP.plasma.ν_en_iz[inw] .== 0.0)
    @test minimum(RP.plasma.ν_ei_eff[inw]) > 0.0

    # The electron half is backward Euler (unconditionally stable) and the ion half is
    # explicit with the tiny rate (me/mi)·nu_ei, so dt = tau is comfortably stable. The
    # relative drift then shrinks by (1 - dt·(me/mi)·nu)/(1 + dt·nu) ~ 1/2 per step, i.e.
    # this relaxation needs tens of steps, not thousands.
    τ = 1 / (sum(RP.plasma.ν_ei_eff[inw]) / length(inw))
    RP.dt = τ

    me, mi = RP.config.constants.me, RP.config.constants.mi
    ue0, ui0 = 1.0e5, 0.0
    RP.plasma.ue_para .= ue0
    RP.plasma.ui_para .= ui0
    p0 = me * n0 * ue0 + mi * n0 * ui0
    u_cm = (me * ue0 + mi * ui0) / (me + mi)

    # The relative drift has to fall well BELOW u_cm (~27 m/s here) before u_e itself can
    # be said to have landed on it; at ~1/2 per step, 60 steps leaves nothing of it.
    for _ in 1:60
        RAPID2D.update_ue_para!(RP)
        RAPID2D.update_ui_para!(RP)
    end

    ue = sum(RP.plasma.ue_para[inw]) / length(inw)
    ui = sum(RP.plasma.ui_para[inw]) / length(inw)
    p1 = me * n0 * ue + mi * n0 * ui

    # The mixed BE(electron)/explicit(ion) pair conserves momentum EXACTLY for any dt,
    # because the ion half is charged with the already-updated u_e — so this is a
    # machine-precision statement, not a discretisation-error one.
    @test isapprox(p1, p0; rtol = 1.0e-10)         # momentum conserved
    @test isapprox(ue, u_cm; rtol = 1.0e-3)        # electrons fell onto u_cm
    @test isapprox(ui, u_cm; rtol = 1.0e-3)        # ions rose onto u_cm
    @test abs(ue - ui) < 1.0e-6 * abs(ue0 - ui0)   # the relative drift is gone
end

@testitem "Physics is invariant to the Ampère routing, for Coulomb on AND off" setup = [PhysicsFixtures] begin
    # Ampère's law is a FIELD solver, not a drag channel. When the plasma current is
    # negligible it must not change the physics, whichever routing workflows.jl picks:
    #
    #   Ampere=false                    -> update_ue_para!
    #   Ampere=true,  |I_tor| < thresh  -> update_ue_para!          (same path)
    #   Ampere=true,  |I_tor| >= thresh -> solve_combined_momentum_Ampere_...!
    #
    # Run as a 2x3 matrix (Coulomb on/off) x (three routings). Two things are asserted:
    # every routing agrees within a Coulomb setting, AND each setting lands on its OWN
    # analytic force balance  u = qe·E∥/(me·(nu_mom + nu_iz [+ nu_ei_eff])) — so the two
    # Coulomb settings are each verified to be RIGHT, not merely mutually consistent.
    #
    # NOTE on the density: this item runs at LOW ne on purpose. The inductive back-EMF
    # and nu_ei both scale with ne, and the induced/applied field ratio is independent of
    # E∥, so "Ampère negligible" and "Coulomb dominant" cannot be arranged together by
    # tuning ne or E. The Coulomb-DOMINATED drag is therefore pinned by the force-balance,
    # decay and momentum-conservation items above; this one pins routing invariance.
    FT = Float64

    function build(; ampere::Bool, threshold::FT, coulomb::Bool)
        config = SimulationConfig{FT}(
            device_Name = "manual",
            NR = 12, NZ = 12,
            prefilled_gas_pressure = 5.0e-3,
            R0B0 = 1.0,
            dt = 1.0e-6,
        )
        config.Output_path = scratch_output_dir()
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(
            ud_evolve = true, ud_method = "Xsec", Implicit = true,
            Atomic_Collision = true,
            Coulomb_Collision = coulomb, Spitzer_Resistivity = true,
            src = false,
            Te_evolve = false, Ti_evolve = false, Gas_evolve = false,
            diffu = false, convec = false,
            Ampere = ampere,
            Ampere_Itor_threshold = threshold,
            E_para_self_ES = false, E_para_self_EM = true,
            update_ni_independently = false,
            Include_ud_convec_term = false,
            Include_ud_pressure_term = false,
            Include_ud_diffu_term = false,
        )
        initialize!(RP)
        RP.plasma.ne .= 1.0e10          # negligible current => negligible self-field
        RP.plasma.ni .= 1.0e10
        RP.plasma.Te_eV .= 10.0
        RP.plasma.Ti_eV .= 1.0
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end

    routings = [
        ("Ampere off", false, 1.0),
        ("below threshold", true, 1.0e9),
        ("coupled solver", true, 0.0),
    ]

    for coulomb in (true, false)
        @testset "Coulomb_Collision = $coulomb" begin
            runs = [
                (name, build(; ampere = amp, threshold = thr, coulomb = coulomb))
                    for (name, amp, thr) in routings
            ]
            for _ in 1:100, (_, RP) in runs
                RAPID2D.advance_timestep!(RP)
            end

            ref = runs[1][2]
            inw = ref.G.nodes.in_wall_nids
            qe, me = ref.config.constants.qe, ref.config.constants.me

            @test maximum(abs, ref.plasma.ue_para[inw]) > 0.0
            I_tor = sum(runs[3][2].plasma.Jϕ) * ref.G.dR * ref.G.dZ
            @test abs(I_tor) < 1.0e-2                 # self-field really is negligible

            # each Coulomb setting lands on ITS OWN analytic balance
            ν_drag = @. ref.plasma.ν_en_mom_tot + ref.plasma.ν_en_iz + ref.plasma.ν_ei_eff
            expected = @. qe * ref.fields.E_para_tot / (me * ν_drag)
            @test isapprox(ref.plasma.ue_para[inw], expected[inw]; rtol = 1.0e-3)

            # ...and every routing agrees with it
            for (name, RP) in runs[2:end]
                @test isapprox(RP.plasma.ue_para[inw], ref.plasma.ue_para[inw]; rtol = 1.0e-3)
            end
        end
    end
end
