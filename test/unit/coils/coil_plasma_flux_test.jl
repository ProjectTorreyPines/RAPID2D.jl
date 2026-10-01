# The plasma flux each coil remembers, ψ_pla(r_c): the value of ψ_pla at the coil that the
# circuit used in its last update. The circuit's d/dt[2π ψ_pla(r_c)] is the difference from
# it, so whatever changes the plasma current between two updates (ionization, transport,
# losses, motion) reaches the coil at the next one.

@testsnippet CoilFluxColumn begin
    # A small column (n = 1e16 m⁻³, Te = 1 eV) in a pure toroidal field with Eϕ = 0.3 V/m at
    # the mean R, Ampère from the first step, and toroidal loops given as (r, z, R, V, name).
    function column_with_loops(loops; t_end = 100.0e-6)
        FT = Float64
        config = SimulationConfig{FT}(;
            device_Name = "manual", manual = ManualSetup{FT}(BR = 0.0, BZ = 0.0, Eϕ = 0.3),
            NR = 20, NZ = 30, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
            dt = 5.0e-6, t_end_s = t_end, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = 100.0e-6,
            Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(;
            Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM = true, ud_evolve = true,
            Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
            Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
            Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
            E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false, FLF_nstep = 100_000,
        )
        initialize!(RP)
        r = @. sqrt((RP.G.R2D - 1.5)^2 + RP.G.Z2D^2)
        RP.plasma.ne .= ifelse.(r .< 0.3, 1.0e16, 0.0)
        RP.plasma.ni .= RP.plasma.ne
        fill!(RP.plasma.Te_eV, 1.0)
        return add_loops!(RP, loops)
    end

    function add_loops!(RP, loops)
        μ0 = RP.config.constants.μ0
        for (rc, zc, R, V, name) in loops
            coil = Coil{Float64}(;
                location = (r = rc, z = zc), area = π * 0.05^2, resistance = R,
                self_inductance = μ0 * rc * (log(8rc / 0.05) - 7 / 4), is_powered = V != 0,
                is_controllable = false, name, voltage_ext = V,
            )
            add_coil!(RP.coil_system, coil)
        end
        initialize_coil_system!(RP)
        return RP
    end

    # the density grows by 1 % after every step; the collision rates follow
    function grow!(rp)
        rp.plasma.ne .*= 1.01
        rp.plasma.ni .*= 1.01
        RAPID2D.update_transport_quantities!(rp)
        return nothing
    end

    quiet(f) = redirect_stdout(f, devnull)
end

@testitem "Coil.ψ_pla starts unset and takes vector assignment" setup = [CoilFactories] begin
    c1, c2 = pf_coil("PF1"), wall_coil("W1")
    @test isnan(c1.ψ_pla) && isnan(c2.ψ_pla)
    csys = CoilSystem([c1, c2])
    csys.coils.ψ_pla = [1.0e-3, -2.0e-3]
    @test csys.coils.ψ_pla == [1.0e-3, -2.0e-3]
end

@testitem "Coupled step: the circuit's flux balance closes every step" setup = [CoilFluxColumn] begin
    # M (Iⁿ⁺¹ − Iⁿ) + Δt R Iⁿ⁺¹ + 2π (ψ_plaⁿ⁺¹ − ψ_plaⁿ) = Δt V, to round-off, whatever the
    # plasma does between steps: here its density grows by 1 % after every step.
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH")])
    rec = (I = Vector{Float64}[], ψ = Vector{Float64}[])
    quiet() do
        run_simulation!(
            RP; callback_after_step = rp -> begin
                push!(rec.I, copy(rp.coil_system.coils.current))
                push!(rec.ψ, copy(rp.coil_system.coils.ψ_pla))
                grow!(rp)
            end
        )
    end
    csys = RP.coil_system
    M, r_c, V, dt = csys.mutual_inductance, get_all_resistances(csys), get_all_voltages_at_time(csys), RP.dt
    residual(k) = M * (rec.I[k] - rec.I[k - 1]) + dt * r_c .* rec.I[k] + 2π * (rec.ψ[k] - rec.ψ[k - 1]) - dt * V
    scale(k) = maximum(abs, M * rec.I[k]) + dt * maximum(abs, V)

    @test length(rec.I) == 20
    @test maximum(maximum(abs, residual(k)) / scale(k) for k in 2:length(rec.I)) < 1.0e-10
end

@testitem "Coupled step: a run split in two is bit-identical" setup = [CoilFluxColumn] begin
    # The coils' memory is state: resuming a run must not recompute it.
    loops = [(1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH")]
    whole = column_with_loops(loops)
    quiet(() -> run_simulation!(whole; callback_after_step = grow!))
    split = column_with_loops(loops; t_end = 50.0e-6)
    quiet(() -> run_simulation!(split; callback_after_step = grow!))
    split.t_end_s = 100.0e-6
    quiet(() -> run_simulation!(split; callback_after_step = grow!))

    @test split.step == whole.step
    @test split.coil_system.coils.current == whole.coil_system.coils.current
    @test split.coil_system.coils.ψ_pla == whole.coil_system.coils.ψ_pla
    @test split.fields.ψ_self == whole.fields.ψ_self
    @test split.plasma.ue_para == whole.plasma.ue_para
end

@testitem "Coupled step: a loop added mid-run starts its own account" setup = [CoilFluxColumn] begin
    # Two superconducting loops: each keeps M I + 2π ψ_pla. Adding the second mid-run must leave
    # the first one's memory alone; the new one takes the plasma flux at its first update.
    RP = column_with_loops([(1.2, 0.8, 1.0e-12, 0.0, "first")]; t_end = 50.0e-6)
    quiet(() -> run_simulation!(RP; callback_after_step = grow!))
    I1, ψ1 = RP.coil_system.coils.current[1], RP.coil_system.coils.ψ_pla[1]
    add_loops!(RP, [(1.8, -0.8, 1.0e-12, 0.0, "second")])

    @test RP.coil_system.coils.ψ_pla[1] == ψ1
    @test isnan(RP.coil_system.coils.ψ_pla[2])

    rec = (I = Vector{Float64}[], ψ = Vector{Float64}[])
    RP.t_end_s = 100.0e-6
    quiet() do
        run_simulation!(
            RP; callback_after_step = rp -> begin
                push!(rec.I, copy(rp.coil_system.coils.current))
                push!(rec.ψ, copy(rp.coil_system.coils.ψ_pla))
                grow!(rp)
            end
        )
    end
    M = RP.coil_system.mutual_inductance
    flux(k) = M * rec.I[k] + 2π * rec.ψ[k]
    before = M[1, 1] * I1 + 2π * ψ1   # the first loop's flux when the second was added (at 0 A)
    # The fluxes themselves can be zero (a loop that started with the plasma at rest), so the
    # tolerance is set by the size of the terms that cancel.
    terms(k) = maximum(abs, M * rec.I[k]) + 2π * maximum(abs, rec.ψ[k])

    @test abs(flux(1)[1] - before) < 1.0e-10 * (abs(M[1, 1] * I1) + abs(2π * ψ1))
    @test maximum(maximum(abs, flux(k) - flux(1)) / terms(k) for k in eachindex(rec.I)) < 1.0e-10
end

@testitem "Coupled step: a step taken without run_simulation! sets the memory" setup = [CoilFluxColumn] begin
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop")])
    RAPID2D.update_transport_quantities!(RP)
    @test all(isnan, RP.coil_system.coils.ψ_pla)
    quiet(() -> advance_timestep!(RP))
    @test all(isfinite, RP.coil_system.coils.ψ_pla)
end
