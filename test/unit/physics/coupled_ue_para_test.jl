# The coupled step solves u∥'s equation as update_ue_para! does, term by term and with the same
# θ-weight (θ_imp.decay): given the field the step induces, the two give the same u∥ however
# strong the coupling. Where the coupling is weak, Ampère on or off then makes no difference to
# u∥.

@testsnippet DiffusiveColumn begin
    # A Gaussian column (1/e radius 0.3 m at R = 1.5 m, Te = 10 eV) in the default manual field,
    # whose weak BZ gives u∥ a pressure term, advection and a parallel part of the diffusivity.
    # u∥ is peaked, and Dperp0 = 500 m²/s diffuses it visibly in one step (D Δt/Δx² ≈ 0.4). One
    # passive loop sits outside the grid. θ is u∥'s θ-weight; with `implicit = false` advection
    # and diffusion are explicit and θ weights the friction only.
    function diffusive_column(; n0, θ = 1.0, implicit = true, ampere = true, diffusion = true, nsteps = 1)
        FT = Float64
        dt = 5.0e-6
        config = SimulationConfig{FT}(;
            device_Name = "manual", manual = ManualSetup{FT}(BR = 0.0, Eϕ = 0.3),
            NR = 20, NZ = 30, R0B0 = 3.0, prefilled_gas_pressure = 0.0, Dperp0 = 500.0,
            dt, t_end_s = nsteps * dt, snap0D_Δt_s = nsteps * dt, snap2D_Δt_s = nsteps * dt,
            Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(;
            Ampere = ampere, Ampere_Itor_threshold = 0.0, E_para_self_EM = true, ud_evolve = true,
            Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
            Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
            Include_ud_convec_term = true, Include_ud_pressure_term = true, Include_ud_diffu_term = diffusion,
            E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false, FLF_nstep = 100_000,
            Implicit = implicit,
        )
        RP.flags.θ_imp.decay = θ
        initialize!(RP)
        μ0 = RP.config.constants.μ0
        add_coil!(
            RP.coil_system, Coil{FT}(;
                location = (r = 0.6, z = 0.0), area = π * 0.05^2, resistance = 1.0e-4,
                self_inductance = μ0 * 0.6 * (log(8 * 0.6 / 0.05) - 7 / 4),
                is_powered = false, is_controllable = false, name = "loop",
            )
        )
        initialize_coil_system!(RP)
        r² = @. (RP.G.R2D - 1.5)^2 + RP.G.Z2D^2
        @. RP.plasma.ne = n0 * exp(-r² / 0.3^2)
        RP.plasma.ni .= RP.plasma.ne
        fill!(RP.plasma.Te_eV, 10.0)
        @. RP.plasma.ue_para = -1.0e6 * exp(-r² / 0.2^2)
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end
end

@testitem "Coupled step: its u∥ equation is update_ue_para!'s at any θ, implicit or not, given the field it induces" setup = [DiffusiveColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!, update_ue_para!
    # A dense column (n = 1e17 m⁻³), whose induced field is some 20 times the applied one. The
    # coupled solve's u∥ satisfies its own row of the block exactly for the ψ it accepts;
    # update_ue_para! with that step's induced field, bϕ Eϕ_self, must give the same u∥, and the
    # same friction ledger Rue_ei, to rounding if the two hold the same terms with the same θ,
    # with Implicit on or off. Diffusion off is the control: both paths then drop the term. The
    # circuits stay backward Euler, as below the gate.
    for implicit in (true, false), θ in (0.0, 0.5, 1.0), diffusion in (false, true)
        a, b = (diffusive_column(; n0 = 1.0e17, θ, implicit, diffusion) for _ in 1:2)
        prepare_timestep!(a)
        prepare_timestep!(b)
        u0 = copy(a.plasma.ue_para)
        solve_combined_momentum_Ampere_equations_with_coils!(a)
        @. b.fields.E_para_tot = b.fields.E_para_ext + b.fields.bϕ * a.fields.Eϕ_self
        update_ue_para!(b)

        Δu = a.plasma.ue_para .- u0
        @test maximum(abs, Δu) > 1.0e-3 * maximum(abs, u0)  # the step moves u∥
        @test maximum(abs, a.plasma.ue_para .- b.plasma.ue_para) <= 1.0e-12 * maximum(abs, Δu)
        @test maximum(abs, a.plasma.Rue_ei .- b.plasma.Rue_ei) <= 1.0e-12 * maximum(abs, b.plasma.Rue_ei)
        @test a.coil_system.θimp == 1
    end
end

@testitem "Coupled step: where the coupling is weak, Ampère makes no difference to u∥ at any θ, implicit or not" setup = [DiffusiveColumn] begin
    # A tenuous column (n = 1e10 m⁻³): (a/δ_e)² ≈ 3e-5, so the plasma's own induction, and the
    # loop's current it drives, barely touch u∥. Three steps run with Ampère on from the first
    # (the coupled solve) and with Ampère off (update_ue_para! on the applied field). u∥ is
    # compared inside the wall: outside it ne = 0, so u∥ there carries no current and its
    # pressure term, through ln n, is not finite.
    for implicit in (true, false), θ in (0.0, 0.5, 1.0)
        on = diffusive_column(; n0 = 1.0e10, θ, implicit, ampere = true, nsteps = 3)
        off = diffusive_column(; n0 = 1.0e10, θ, implicit, ampere = false, nsteps = 3)
        inw = on.G.nodes.in_wall_nids
        u0 = on.plasma.ue_para[inw]
        for RP in (on, off)
            redirect_stderr(devnull) do
                redirect_stdout(() -> run_simulation!(RP), devnull)
            end
        end
        @test on.diagnostics.ampere_picard.nsolve == 3
        @test off.diagnostics.ampere_picard.nsolve == 0

        u_on, u_off = on.plasma.ue_para[inw], off.plasma.ue_para[inw]
        @test maximum(abs, u_on .- u_off) <= 1.0e-3 * maximum(abs, u_off .- u0)
    end
end

@testitem "Coupled step: the circuits take θ_imp.circuit on both sides of the gate" setup = [DiffusiveColumn] begin
    # No plasma, and the loop starts at 1 kA: its current decays on its own circuit,
    # I^{n+1} = (L − (1−θ)ΔtR)/(L + θΔtR) I^n, below the gate (the vacuum circuits) and above it
    # (the coupled solve) alike, with θ the circuits' weight whatever u∥'s is.
    for θc in (0.0, 0.5, 1.0), gate in (0.0, Inf)
        RP = diffusive_column(; n0 = 0.0, θ = 0.5, nsteps = 3)
        RP.flags.θ_imp.circuit = θc
        RP.flags.Ampere_Itor_threshold = gate
        set_all_currents!(RP.coil_system, [1.0e3])
        redirect_stderr(devnull) do
            redirect_stdout(() -> run_simulation!(RP), devnull)
        end
        @test RP.diagnostics.ampere_picard.nsolve == (gate == 0 ? 3 : 0)

        c = RP.coil_system.coils[1]
        L, R, dt = c.self_inductance, c.resistance, RP.dt
        g = (L - (1 - θc) * dt * R) / (L + θc * dt * R)
        @test get_all_currents(RP.coil_system)[1] ≈ 1.0e3 * g^3 rtol = 1.0e-12
    end
end

@testitem "update_ue_para!: Implicit = false is its θ = 0 case, every term included" setup = [DiffusiveColumn] begin
    using RAPID2D: update_ue_para!
    # Explicit stepping is the θ-scheme at θ = 0: advection, pressure and diffusion from uⁿ, and
    # the update solved without a matrix. With the friction at θ = 0 too, it is the implicit
    # update at θ = 0, to rounding.
    a = diffusive_column(; n0 = 1.0e17, θ = 0.0, implicit = false)
    b = diffusive_column(; n0 = 1.0e17, θ = 0.0, implicit = true)
    u0 = copy(a.plasma.ue_para)
    for RP in (a, b)
        prepare_timestep!(RP)
        update_ue_para!(RP)
    end
    Δu = b.plasma.ue_para .- u0
    @test maximum(abs, Δu) > 1.0e-3 * maximum(abs, u0)
    @test maximum(abs, a.plasma.ue_para .- b.plasma.ue_para) <= 1.0e-12 * maximum(abs, Δu)
    @test maximum(abs, a.plasma.Rue_ei .- b.plasma.Rue_ei) <= 1.0e-12 * maximum(abs, b.plasma.Rue_ei)
end

# ── The near-vacuum relaxation ────────────────────────────────────────────────
#
# Where there is almost no plasma, u∥ and Te are relaxed implicitly (u∥ → 0, Te → the local
# plasma's Te). It must sit in both u∥ paths, so the gate equivalence above still holds, and it
# must leave a plasma that is everywhere denser than 100·n_vac exactly as it was.

@testitem "near-vacuum relaxation: the weight is (n_vac/n)² below 10·n_vac and exactly zero from 100·n_vac" begin
    using RAPID2D: vacuum_relaxation_weight
    n_vac = 1.0e4
    @test vacuum_relaxation_weight(1.0e2, n_vac) == 1.0e4
    @test vacuum_relaxation_weight(n_vac, n_vac) == 1.0
    @test vacuum_relaxation_weight(0.0, n_vac) == 1.0e12          # capped, also at n = 0
    @test vacuum_relaxation_weight(1.0e-9, n_vac) == 1.0e12
    @test all(vacuum_relaxation_weight(n, n_vac) == 0 for n in (1.0e6, 3.0e6, 1.0e17))
    # continuous where the fade starts and ends, and decreasing in n
    @test vacuum_relaxation_weight(10n_vac * (1 + 1.0e-9), n_vac) ≈ vacuum_relaxation_weight(10n_vac, n_vac) rtol = 1.0e-6
    @test vacuum_relaxation_weight(100n_vac * (1 - 1.0e-6), n_vac) < 1.0e-14
    ns = 10.0 .^ range(-3, 6; length = 400)
    @test issorted(vacuum_relaxation_weight.(ns, n_vac); rev = true)
end

@testitem "near-vacuum relaxation: a plasma denser than 100·n_vac everywhere is untouched, bit for bit" setup = [DiffusiveColumn] begin
    using RAPID2D: update_ue_para!, update_Te!
    # The column's thinnest in-wall node is ~1e11 m⁻³, far above 100·n_vac
    for implicit in (true, false)
        on, off = (diffusive_column(; n0 = 1.0e17, implicit) for _ in 1:2)
        off.flags.vacuum_relaxation = (state = false, factor = on.flags.vacuum_relaxation.factor)
        @test minimum(on.plasma.ne[on.G.nodes.in_wall_nids]) > 100 * on.flags.vacuum_relaxation.factor
        for RP in (on, off)
            prepare_timestep!(RP)
            update_ue_para!(RP)
            update_Te!(RP)
        end
        @test on.plasma.ue_para == off.plasma.ue_para
        @test on.plasma.Te_eV == off.plasma.Te_eV
    end
end

@testitem "near-vacuum relaxation: u∥ on a near-empty patch goes to the u∥ around it, the same in update_ue_para! and in the coupled step" setup = [DiffusiveColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!, update_ue_para!,
        vacuum_relaxation_weights, vacuum_reference, ExpRB
    # An almost empty patch (1e-2 m⁻³) inside the dense column, where the pressure force on the
    # few particles is enormous. The gate equivalence must survive it, and u∥ there must follow
    # the density-weighted mean of its neighbours at the start of the step.
    function with_hole(; implicit)
        RP = diffusive_column(; n0 = 1.0e17, implicit)
        G = RP.G
        hole = [k for k in G.nodes.in_wall_nids if 0.25^2 < (G.R2D[k] - 1.5)^2 + G.Z2D[k]^2 < 0.35^2 && G.Z2D[k] > 0]
        RP.plasma.ne[hole] .= 1.0e-2
        RP.plasma.ni[hole] .= 1.0e-2
        RAPID2D.update_transport_quantities!(RP)
        return RP, hole
    end
    reference(RP) = vacuum_reference(RP, RP.plasma.ne, RP.plasma.ue_para, vacuum_relaxation_weights(RP, RP.plasma.ne))
    for implicit in (true, false)
        (a, hole), (b, _) = with_hole(; implicit), with_hole(; implicit)
        @test length(hole) > 5
        prepare_timestep!(a)
        prepare_timestep!(b)
        u0, u_ref = copy(a.plasma.ue_para), reference(a)
        @test minimum(abs, u_ref[hole]) > 1.0e-3 * maximum(abs, u0)       # a reference other than 0
        solve_combined_momentum_Ampere_equations_with_coils!(a)
        @. b.fields.E_para_tot = b.fields.E_para_ext + b.fields.bϕ * a.fields.Eϕ_self
        update_ue_para!(b)
        Δu = a.plasma.ue_para .- u0
        @test maximum(abs, a.plasma.ue_para .- b.plasma.ue_para) <= 1.0e-12 * maximum(abs, Δu)
        @test maximum(abs, a.plasma.ue_para[hole] .- u_ref[hole]) <= 1.0e-6 * maximum(abs, u0)
        @test maximum(abs, a.plasma.ue_para[hole]) <= maximum(abs, u0)    # within the neighbours' values
    end
    # update_ue_para! under ExpRB on the friction, with and without the matrix
    for implicit in (true, false)
        RP, hole = with_hole(; implicit)
        RP.flags.scheme.decay = ExpRB
        prepare_timestep!(RP)
        u0, u_ref = copy(RP.plasma.ue_para), reference(RP)
        update_ue_para!(RP)
        @test maximum(abs, RP.plasma.ue_para[hole] .- u_ref[hole]) <= 1.0e-6 * maximum(abs, u0)
    end
end

@testitem "near-vacuum relaxation: Te on a near-empty patch goes to the Te of the plasma around it" setup = [DiffusiveColumn] begin
    using RAPID2D: update_Te!
    for implicit in (true, false)
        RP = diffusive_column(; n0 = 1.0e17, implicit)
        G = RP.G
        # thinner than a cell, so every node of it has plasma next to it
        hole = [k for k in G.nodes.in_wall_nids if 0.28^2 < (G.R2D[k] - 1.5)^2 + G.Z2D[k]^2 < 0.32^2 && G.Z2D[k] > 0]
        @test length(hole) > 2
        RP.plasma.ne[hole] .= 1.0e-2
        RP.plasma.Te_eV[hole] .= 400.0          # the plasma around it is at 10 eV
        RAPID2D.update_transport_quantities!(RP)
        prepare_timestep!(RP)
        update_Te!(RP)
        @test all(abs.(RP.plasma.Te_eV[hole] .- 10.0) .< 1.0)
    end
end

@testitem "near-vacuum ledger: what the relaxation exchanged, the floorings, the snapshot, the gate retry" setup = [DiffusiveColumn] begin
    using RAPID2D: update_ue_para!, update_Te!, measure_snap0D!, vacuum_relaxation_weights,
        correct_negative_densities!, below_gate_entry_state, restore_below_gate_entry_state!
    # a dense column books nothing
    RP = diffusive_column(; n0 = 1.0e17)
    prepare_timestep!(RP)
    update_ue_para!(RP)
    update_Te!(RP)
    @test RP.diagnostics.near_vacuum.relax_momentum == 0
    @test RP.diagnostics.near_vacuum.relax_energy == 0

    # a near-empty patch at 400 eV: something is exchanged, nothing next to the plasma's own
    RP = diffusive_column(; n0 = 1.0e17)
    G = RP.G
    hole = [k for k in G.nodes.in_wall_nids if 0.28^2 < (G.R2D[k] - 1.5)^2 + G.Z2D[k]^2 < 0.32^2 && G.Z2D[k] > 0]
    RP.plasma.ne[hole] .= 1.0e-2
    RP.plasma.Te_eV[hole] .= 400.0
    RAPID2D.update_transport_quantities!(RP)
    prepare_timestep!(RP)
    update_ue_para!(RP)
    update_Te!(RP)
    nv = RP.diagnostics.near_vacuum
    me, ee = RP.config.constants.me, RP.config.constants.ee
    V, n = G.inVol2D, RP.plasma.ne
    momentum_scale = sum(@. V * n * me * abs(RP.plasma.ue_para))
    energy_scale = sum(@. V * n * 1.5 * ee * RP.plasma.Te_eV)
    @test nv.relax_momentum != 0 && abs(nv.relax_momentum) < 1.0e-10 * momentum_scale
    @test 0 < nv.relax_energy < 1.0e-10 * energy_scale       # 400 eV pulled down to the plasma's 10 eV

    # the snapshot counts the relaxed nodes and reads the ledger
    snap = Snapshot0D{Float64}()
    measure_snap0D!(RP, snap)
    @test snap.vac_relax_nodes == count(>(0), vacuum_relaxation_weights(RP, n)) >= length(hole)
    @test 0 < snap.vac_relax_Ne_share < 1.0e-15
    @test (snap.vac_relax_momentum, snap.vac_relax_energy) == (nv.relax_momentum, nv.relax_energy)

    # a step solved again above the gate puts the ledger back
    entry = below_gate_entry_state(RP)
    nv.relax_momentum += 1.0
    nv.floor_nodes += 3
    restore_below_gate_entry_state!(RP, entry)
    @test RP.diagnostics.near_vacuum.relax_momentum == snap.vac_relax_momentum
    @test RP.diagnostics.near_vacuum.floor_nodes == 0

    # the floor counts what it floors, and the electrons it creates
    floored = hole[1:3]
    RP.plasma.ne[floored] .= -1.0
    correct_negative_densities!(RP)
    created = sum((RP.plasma.ne[k] + 1.0) * 2π * G.Jacob[k] * G.dR * G.dZ for k in floored)
    @test RP.diagnostics.near_vacuum.floor_nodes == 3
    @test RP.diagnostics.near_vacuum.floor_Ne ≈ created
end

@testitem "near-vacuum relaxation: a run lowers an n_vac that would reach its initial plasma" setup = [DiffusiveColumn] begin
    using RAPID2D: check_vacuum_relaxation_density!
    # n̄ = Σ J n² / Σ J n over the in-wall nodes, the density an average electron sits in
    function mean_density(RP)
        IN = RP.G.nodes.in_wall_nids
        n, J = RP.plasma.ne[IN], RP.G.Jacob[IN]
        return sum(J .* n .^ 2) / sum(J .* n)
    end

    # the default n_vac against a dense column: untouched
    RP = diffusive_column(; n0 = 1.0e17)
    vr = RP.flags.vacuum_relaxation
    @test_logs check_vacuum_relaxation_density!(RP)
    @test RP.flags.vacuum_relaxation === vr

    # 1e4 against a ~1e6 column: the relaxation would end inside it, so n_vac goes to n̄/1e4
    RP = diffusive_column(; n0 = 1.0e6)
    RP.flags.vacuum_relaxation = (state = true, factor = 1.0e4)
    @test_logs (:warn, r"lowered to") check_vacuum_relaxation_density!(RP)
    @test RP.flags.vacuum_relaxation.state
    @test RP.flags.vacuum_relaxation.factor ≈ mean_density(RP) / 1.0e4

    # off, or no plasma: nothing to check
    RP.flags.vacuum_relaxation = (state = false, factor = 1.0e4)
    @test_logs check_vacuum_relaxation_density!(RP)
    @test RP.flags.vacuum_relaxation.factor == 1.0e4
    RP.flags.vacuum_relaxation = (state = true, factor = 1.0e4)
    fill!(RP.plasma.ne, 0.0)
    @test_logs check_vacuum_relaxation_density!(RP)
    @test RP.flags.vacuum_relaxation.factor == 1.0e4

    # run_simulation! checks before its first step
    RP = diffusive_column(; n0 = 1.0e6)
    RP.flags.vacuum_relaxation = (state = true, factor = 1.0e4)
    n̄ = mean_density(RP)
    @test_logs (:warn, r"lowered to") match_mode = :any redirect_stdout(() -> run_simulation!(RP), devnull)
    @test RP.step == 1
    @test RP.flags.vacuum_relaxation.factor ≈ n̄ / 1.0e4
end
