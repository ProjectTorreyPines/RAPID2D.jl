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
