# The plasma flux each coil remembers, ψ_pla(r_c): the value of ψ_pla at the coil that the
# circuit used in its last update. The circuit's d/dt[2π ψ_pla(r_c)] is the difference from
# it, so whatever changes the plasma current between two updates (ionization, transport,
# losses, motion) reaches the coil at the next one.

@testsnippet CoilFluxColumn begin
    # A small column (n = 1e16 m⁻³, Te = 1 eV) in a pure toroidal field with Eϕ = E0 at the
    # mean R, Ampère from the first step, and toroidal loops given as (r, z, R, V, name).
    function column_with_loops(loops; t_end = 100.0e-6, E0 = 0.3)
        FT = Float64
        config = SimulationConfig{FT}(;
            device_Name = "manual", manual = ManualSetup{FT}(BR = 0.0, BZ = 0.0, Eϕ = E0),
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

@testitem "Coupled step: the circuit's flux balance closes on solves cut short" setup = [CoilFluxColumn] begin
    # As above, every coupled solve stopped after two iterations, far from converged: the
    # circuits still close their balance, with the flux of the current each solve accepted.
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH")])
    RP.flags.ampere_picard = PicardSettings{Float64}(; tolerance = 1.0e-300, max_iter = 2)
    rec = (I = Vector{Float64}[], ψ = Vector{Float64}[])
    quiet() do
        redirect_stderr(devnull) do
            run_simulation!(
                RP; callback_after_step = rp -> begin
                    push!(rec.I, copy(rp.coil_system.coils.current))
                    push!(rec.ψ, copy(rp.coil_system.coils.ψ_pla))
                    grow!(rp)
                end
            )
        end
    end
    csys = RP.coil_system
    M, r_c, V, dt = csys.mutual_inductance, get_all_resistances(csys), get_all_voltages_at_time(csys), RP.dt
    residual(k) = M * (rec.I[k] - rec.I[k - 1]) + dt * r_c .* rec.I[k] + 2π * (rec.ψ[k] - rec.ψ[k - 1]) - dt * V
    scale(k) = maximum(abs, M * rec.I[k]) + dt * maximum(abs, V)

    @test RP.diagnostics.ampere_picard.nunconverged == RP.diagnostics.ampere_picard.nsolve
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

@testitem "Coils that join a run late keep its time, on every step path" setup = [CoilFluxColumn] begin
    # A run without coils, then a coil whose voltage ramps, V(t) = 10 V + 10⁶ V/s · t, set up at
    # 50 µs. The coils' clock, which the 0D snapshots and the calls without a time read, must
    # read the run's time from then on, below the gate, in the split step and in the coupled
    # solve: every snapshot records the voltage at its own time.
    V(t) = 10.0 + 1.0e6 * t
    for (threshold, ud_evolve) in ((1.0e9, true), (0.0, false), (0.0, true))
        RP = column_with_loops([]; t_end = 50.0e-6)
        RP.flags.Ampere_Itor_threshold = threshold
        RP.flags.ud_evolve = ud_evolve
        quiet(() -> run_simulation!(RP))
        add_loops!(RP, [(0.6, 0.0, 1.0e-4, V, "OH")])
        @test RP.coil_system.time_s ≈ RP.time_s

        nsnap = length(RP.diagnostics.snaps0D)
        RP.t_end_s = 100.0e-6
        quiet(() -> run_simulation!(RP))
        snaps = RP.diagnostics.snaps0D[(nsnap + 1):end]
        @test RP.coil_system.time_s ≈ RP.time_s
        @test length(snaps) == 2   # the resumed run's first snapshot, and the one at 100 µs
        @test all(s.coils_V_ext[1] ≈ V(s.time_s) for s in snaps)
    end
end

@testitem "Coupled step: a step taken without run_simulation! sets the memory" setup = [CoilFluxColumn] begin
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop")])
    RAPID2D.update_transport_quantities!(RP)
    @test all(isnan, RP.coil_system.coils.ψ_pla)
    quiet(() -> advance_timestep!(RP))
    @test all(isfinite, RP.coil_system.coils.ψ_pla)
end

@testitem "Coupled step: with E_para_self_EM off the coils induce no Eϕ_self" setup = [CoilFluxColumn] begin
    # Below the gate, a 10 V coil changes; with the electromagnetic self-field off nothing is
    # induced on the plasma, as before the coils advanced there.
    RP = column_with_loops([(0.6, 0.0, 1.0e-4, 10.0, "OH")]; t_end = 25.0e-6)
    RP.flags.E_para_self_EM = false
    RP.flags.Ampere_Itor_threshold = 1.0e6
    quiet(() -> run_simulation!(RP))
    @test abs(RP.coil_system.coils.current[1]) > 0
    @test all(iszero, RP.fields.Eϕ_self)
end

@testitem "Coupled step: the coils' memory starts from the current ψ_self is solved from" setup = [CoilFluxColumn] begin
    using RAPID2D: plasma_flux_at_coils
    # A column carrying ~10 kA at the start. The initial Grad–Shafranov solve sources ψ_self
    # with J₀; adding that self field tilts b, so J moves after it. The coils must start
    # from J₀ as ψ_self does, or the first step reads the difference as induced on one side
    # only.
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop")])
    in_column = RP.plasma.ne .> 0
    RP.plasma.ne[in_column] .= 1.0e18
    RP.plasma.ni .= RP.plasma.ne
    RP.plasma.ue_para[in_column] .= -2.5e5
    RAPID2D.update_Jϕ!(RP)
    J0 = copy(RP.plasma.Jϕ)
    initialize_coupled_fields!(RP)

    @test RP.plasma.Jϕ != J0   # b moved: the test can tell J₀ from the J after it
    ψ0 = plasma_flux_at_coils(RP.coil_system, RP.G, J0)
    @test maximum(abs, RP.coil_system.coils.ψ_pla .- ψ0) <= 1.0e-14 * maximum(abs, ψ0)
end

@testitem "Coupled step: below the gate a resistive loop decays as in vacuum" setup = [CoilFluxColumn] begin
    # Below the gate the plasma current is not a source of induction: the loop does not see
    # it, and the column is driven by the loop's decay alone. A dense, hot column at rest, no
    # applied field, beside a pre-charged loop with Δt R/L ≈ 0.07, the gate held above
    # anything reached. Driven so, with fixed collision rates, the column's current follows
    # the loop's in a fixed ratio once its 1/(1 + νΔt) transient is gone. Were the loop's
    # reaction fed back a step late, through its resistive decay, the pair would grow without
    # bound and flip sign every other step.
    RP = column_with_loops([(1.5, 0.45, 0.1, 0.0, "loop")]; t_end = 150.0e-6, E0 = 0.0)
    in_column = RP.plasma.ne .> 0
    RP.plasma.ne[in_column] .= 1.0e18
    RP.plasma.ni .= RP.plasma.ne
    fill!(RP.plasma.Te_eV, 10.0)
    RP.flags.Ampere_Itor_threshold = 1.0e9
    RP.coil_system.coils.current = [100.0]
    RAPID2D.update_transport_quantities!(RP)
    Ip, Ic = Float64[], Float64[]
    quiet() do
        run_simulation!(
            RP; callback_after_step = rp -> begin
                push!(Ip, sum(rp.plasma.Jϕ) * rp.G.dR * rp.G.dZ)
                push!(Ic, rp.coil_system.coils.current[1])
            end
        )
    end
    L, R = RP.coil_system.mutual_inductance[1, 1], get_all_resistances(RP.coil_system)[1]
    vacuum = [100.0 * (L / (L + RP.dt * R))^k for k in eachindex(Ic)]
    ratio = [Ip[k] / Ic[k - 1] for k in 8:length(Ip)]

    @test length(Ic) == 30
    @test maximum(abs, Ic .- vacuum) <= 1.0e-10 * 100.0
    @test all(>(0), Ip)
    @test maximum(ratio) - minimum(ratio) <= 1.0e-3 * minimum(ratio)
end

@testitem "Coupled step: steps taken before run_simulation! keep the coils' memory" setup = [CoilFluxColumn] begin
    # A few steps by advance_timestep!, then run_simulation!. The run must not reset the
    # memory those steps left (RP.step is still 0): the flux balance closes across the switch.
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop")]; t_end = 50.0e-6)
    RAPID2D.update_transport_quantities!(RP)
    quiet() do
        for _ in 1:3
            advance_timestep!(RP)
            RP.time_s += RP.dt
            grow!(RP)
        end
    end
    I0, ψ0 = copy(RP.coil_system.coils.current), copy(RP.coil_system.coils.ψ_pla)
    rec = (I = Vector{Float64}[], ψ = Vector{Float64}[])
    quiet() do
        run_simulation!(
            RP; callback_after_step = rp -> begin
                push!(rec.I, copy(rp.coil_system.coils.current))
                push!(rec.ψ, copy(rp.coil_system.coils.ψ_pla))
            end
        )
    end
    csys = RP.coil_system
    M, r_c, dt = csys.mutual_inductance, get_all_resistances(csys), RP.dt
    residual = M * (rec.I[1] - I0) + dt * r_c .* rec.I[1] + 2π * (rec.ψ[1] - ψ0)
    @test maximum(abs, residual) < 1.0e-10 * (maximum(abs, M * rec.I[1]) + 2π * maximum(abs, rec.ψ[1]))
end

@testitem "Split step: above the gate without the coupled solve, the circuits take the plasma's flux change" setup = [CoilFluxColumn] begin
    # With ud_evolve off the coupled solve does not run, and the coils advance on their own
    # circuits with the plasma term, the change from the flux each coil remembers. A column
    # with a fixed drift whose density grows by 1 % after every step: the flux balance must
    # close every step, as it does in the coupled solve.
    RP = column_with_loops([(1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH")])
    RP.flags.ud_evolve = false
    RP.plasma.ue_para[RP.plasma.ne .> 0] .= -1.0e5
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

    @test RP.diagnostics.ampere_picard.nsolve == 0      # the coupled solve never ran
    @test maximum(abs, rec.ψ[end] .- rec.ψ[1]) > 0      # the coils saw the plasma change
    @test maximum(maximum(abs, residual(k)) / scale(k) for k in 2:length(rec.I)) < 1.0e-10
end

@testitem "Split step: the coils step with the run's Δt" setup = [CoilFluxColumn] begin
    # advance_coils! rebuilds the circuit matrices when the run's Δt (or θ) differs from the
    # one they were built with. A resistive loop below the gate, its circuit built for 2Δt:
    # one step must decay it by L/(L + Δt R).
    RP = column_with_loops([(1.2, 0.8, 0.1, 0.0, "loop")])
    csys = RP.coil_system
    csys.coils.current = [100.0]
    csys.Δt = 2 * RP.dt
    RAPID2D.calculate_circuit_matrices!(csys)
    L, R = csys.mutual_inductance[1, 1], get_all_resistances(csys)[1]
    ΔI = RAPID2D.advance_coils!(RP; plasma = false)

    @test csys.Δt == RP.dt
    @test csys.coils.current[1] ≈ 100.0 * L / (L + RP.dt * R) rtol = 1.0e-12
    @test ΔI[1] ≈ csys.coils.current[1] - 100.0 rtol = 1.0e-12
end

@testitem "Coil steps: the coils' clock ends at the step's time plus Δt" setup = [CoilFluxColumn] begin
    using RAPID2D: advance_LR_circuit_step!, solve_combined_momentum_Ampere_equations_with_coils!
    # A coil step is taken at the time it is given, the run's. The coils' clock must end at
    # that time plus Δt whatever it read before; here the run's time is moved past it.
    t = 40.0e-6
    steps = (
        rp -> advance_LR_circuit_step!(rp.coil_system, t),
        rp -> advance_LR_circuit_step!(rp.coil_system, rp.G, rp.plasma.Jϕ, t),
        solve_combined_momentum_Ampere_equations_with_coils!,
        rp -> solve_combined_momentum_Ampere_equations_with_coils!(rp; method = :direct),
    )
    for step! in steps
        RP = column_with_loops([(0.6, 0.0, 1.0e-4, 10.0, "OH")])
        RAPID2D.update_transport_quantities!(RP)
        RP.time_s = t
        prepare_timestep!(RP)
        step!(RP)
        @test RP.coil_system.time_s ≈ t + RP.dt
    end
end

@testitem "Coupled step: the direct solve takes the step the iteration converges to" setup = [CoilFluxColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # Two identical columns with a resistive loop and a powered coil, a few steps in; then one
    # step iterated to a tight tolerance without floors, and one solved directly.
    loops = [(1.2, 0.8, 1.0e-3, 0.0, "loop"), (0.6, 0.0, 1.0e-4, 10.0, "OH")]
    a, b = column_with_loops(loops; t_end = 50.0e-6), column_with_loops(loops; t_end = 50.0e-6)
    for RP in (a, b)
        RAPID2D.update_transport_quantities!(RP)
        quiet(() -> run_simulation!(RP))
        prepare_timestep!(RP)
    end
    tight = (tolerance = 1.0e-12, max_iter = 200, E_floor = 0.0, I_floor = 0.0)
    solve_combined_momentum_Ampere_equations_with_coils!(a; tight...)
    solve_combined_momentum_Ampere_equations_with_coils!(b; tight..., method = :direct)
    agree(x, y) = maximum(abs, x .- y) <= 1.0e-10 * maximum(abs, y)

    @test a.diagnostics.ampere_picard.nunconverged == 0 && b.diagnostics.ampere_picard.nunconverged == 0
    @test agree(b.plasma.ue_para, a.plasma.ue_para)
    @test agree(b.fields.ψ_self, a.fields.ψ_self)
    @test agree(b.fields.Eϕ_self, a.fields.Eϕ_self)
    @test agree(b.coil_system.coils.current, a.coil_system.coils.current)
    @test agree(b.coil_system.coils.ψ_pla, a.coil_system.coils.ψ_pla)
end
