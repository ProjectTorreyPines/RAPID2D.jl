# The coupled solve's Picard stopping test, in the units of the induced field: the last
# iteration's change of E_ind against what the step induces, plus an absolute floor, and the
# coil currents likewise. Counts of solves, iterations and unconverged solves are kept in
# RP.diagnostics.ampere_picard.

@testsnippet PicardColumn begin
    # A column (n0 in m⁻³, Te = 1 eV) in a pure toroidal field with Eϕ = E0 at the mean R,
    # Ampère from the first step.
    function picard_column(; n0 = 1.0e16, E0 = 0.3, t_end = 50.0e-6)
        FT = Float64
        config = SimulationConfig{FT}(;
            device_Name = "manual", manual = ManualSetup{FT}(BR = 0.0, BZ = 0.0, Eϕ = E0),
            NR = 20, NZ = 30, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
            dt = 5.0e-6, t_end_s = t_end, snap0D_Δt_s = 50.0e-6, snap2D_Δt_s = 50.0e-6,
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
        RP.plasma.ne .= ifelse.(r .< 0.3, n0, 0.0)
        RP.plasma.ni .= RP.plasma.ne
        fill!(RP.plasma.Te_eV, 1.0)
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end
end

@testitem "Ampère Picard: a step that induces nothing stops at once" setup = [PicardColumn] begin
    # No plasma and no coils: the field never changes, so one iteration settles each step,
    # with no warning (the old ‖Δψ‖/‖ψ‖ test was 0/0 here and ran to max_iter).
    RP = picard_column(; n0 = 0.0, E0 = 0.0)
    logs, _ = Test.collect_test_logs(min_level = Base.CoreLogging.Warn) do
        redirect_stdout(() -> run_simulation!(RP), devnull)
    end
    @test !any(log -> occursin("Picard", string(log.message)), logs)
    stats = RP.diagnostics.ampere_picard
    @test stats.nsolve == 10
    @test stats.niter == stats.nsolve
    @test stats.nunconverged == 0
end

@testitem "Ampère Picard: an unconverged solve is counted, and warned once per run" setup = [PicardColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    RP = picard_column()
    stats = RP.diagnostics.ampere_picard
    @test_logs (:warn, r"Picard") begin
        for _ in 1:3
            prepare_timestep!(RP)
            solve_combined_momentum_Ampere_equations_with_coils!(RP; tolerance = 1.0e-300, max_iter = 2)
        end
    end
    @test stats.nsolve == 3
    @test stats.nunconverged == 3
    @test stats.niter == 6
    @test stats.last_E_residual > 0 && isfinite(stats.last_I_residual)
end

@testitem "Ampère Picard: the run's settings reach the coupled solve" setup = [PicardColumn] begin
    # flags.ampere_picard is what solve_timestep! hands the coupled solve: a tolerance no
    # iteration can meet makes every solve run exactly its max_iter block solves.
    RP = picard_column(; t_end = 20.0e-6)
    RP.flags.ampere_picard.tolerance = 1.0e-300
    RP.flags.ampere_picard.max_iter = 3
    redirect_stderr(devnull) do
        redirect_stdout(() -> run_simulation!(RP), devnull)
    end
    stats = RP.diagnostics.ampere_picard
    @test stats.nsolve == 4
    @test stats.niter == 3 * stats.nsolve
    @test stats.nunconverged == stats.nsolve
end

@testitem "Ampère Picard: the stopping test bounds the induced-field error" setup = [PicardColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # Two identical columns, a few steps in; then one more step solved with the defaults and
    # with a tolerance tight enough, and no floors, to stand for the exact solution. The
    # default's induced field must agree with it to within the default tolerance (1e-3) of
    # the step's field.
    a, b = picard_column(), picard_column()
    for RP in (a, b)
        redirect_stdout(() -> run_simulation!(RP), devnull)
        prepare_timestep!(RP)
    end
    solve_combined_momentum_Ampere_equations_with_coils!(a)
    solve_combined_momentum_Ampere_equations_with_coils!(b; tolerance = 1.0e-12, max_iter = 200, E_floor = 0.0, I_floor = 0.0)
    E_a, E_b = a.fields.Eϕ_self, b.fields.Eϕ_self

    @test b.diagnostics.ampere_picard.nunconverged == 0
    @test maximum(abs, E_a .- E_b) <= 1.0e-3 * maximum(abs, E_b)
end

@testsnippet BoundaryLimitedColumn begin
    # Dense, hot columns (n = 1e18 m⁻³, Te = 10 eV) on a 20×30 grid where the coupled solve's
    # outer iteration has trouble, Ampère from the first step, Coulomb friction only:
    # - tight_column: radius 0.55 m at (1.6, 0), filling a box wall one cell inside a 1.2 × 1.6 m
    #   grid; its own flux returns through the boundary values with a large overshoot;
    # - shell_column: radius 0.3 m at (1.5, 0) inside 24 copper filaments on a circle of radius
    #   0.35 m, whose eddy currents follow the plasma slowly from one iteration to the next;
    # - mixed_column: as shell_column with the shell at 0.4 m, u∥ advection on, and a powered
    #   coil of 1 mH outside the grid, so that the coils' inductances span four decades.
    function dense_column(; manual, cenR, radius, n0 = 1.0e18, Te = 10.0, t_end = 10.0e-6, advection = false)
        FT = Float64
        config = SimulationConfig{FT}(;
            device_Name = "manual", manual, NR = 20, NZ = 30, R0B0 = 3.0, prefilled_gas_pressure = 0.0,
            dt = 5.0e-6, t_end_s = t_end, snap0D_Δt_s = t_end, snap2D_Δt_s = t_end,
            Output_path = mktempdir(; cleanup = false),
        )
        RP = RAPID{FT}(config)
        RP.flags = SimulationFlags{FT}(;
            Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM = true, ud_evolve = true,
            Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
            Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
            Include_ud_convec_term = advection, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
            E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false, FLF_nstep = 100_000,
        )
        initialize!(RP)
        r = @. sqrt((RP.G.R2D - cenR)^2 + RP.G.Z2D^2)
        RP.plasma.ne .= ifelse.(r .< radius, n0, 0.0)
        RP.plasma.ni .= RP.plasma.ne
        fill!(RP.plasma.Te_eV, Te)
        RAPID2D.update_transport_quantities!(RP)
        return RP
    end

    function add_filament_shell!(RP; cenR, r, nfil = 24)
        μ0 = RP.config.constants.μ0
        side = 2π * r / nfil
        for k in 1:nfil
            θ = 2π * (k - 0.5) / nfil + 0.05
            R, Z = cenR + r * cos(θ), r * sin(θ)
            add_coil!(
                RP.coil_system, Coil{Float64}(;
                    location = (r = R, z = Z), area = side^2, resistance = 1.68e-8 * 2π * R / side^2,
                    self_inductance = μ0 * R * (log(8R / (side / sqrt(π))) - 7 / 4),
                    is_powered = false, is_controllable = false, name = "shell_$k",
                )
            )
        end
        initialize_coil_system!(RP)
        return RP
    end

    tight_column(; kw...) = dense_column(;
        manual = ManualSetup{Float64}(R = (1.0, 2.2), Z = (-0.8, 0.8), BR = 0.0, BZ = 0.0, Eϕ = 0.3, wall_margin_cells = 1),
        cenR = 1.6, radius = 0.55, kw...,
    )
    shell_column(; kw...) = add_filament_shell!(
        dense_column(; manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = 0.3), cenR = 1.5, radius = 0.3, kw...);
        cenR = 1.5, r = 0.35,
    )
    function mixed_column(; kw...)
        RP = dense_column(;
            manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.0, Eϕ = 0.3), cenR = 1.5, radius = 0.3, advection = true, kw...,
        )
        add_coil!(
            RP.coil_system, Coil{Float64}(;
                location = (r = 0.6, z = 0.0), area = π * 0.1^2, resistance = 1.0e-3, self_inductance = 1.0e-3,
                is_powered = true, is_controllable = false, name = "CS", voltage_ext = 100.0,
            )
        )
        return add_filament_shell!(RP; cenR = 1.5, r = 0.4)
    end

    # The same equations solved directly, without the iteration: the map's linear part
    # assembled and Newton's step taken on it, checked to 1e-10 of the step's field and of each
    # coil's change, with no floors.
    const REFERENCE_PICARD = (method = DirectOuterSolve(), tolerance = 1.0e-10, E_floor = 0.0, I_floor = 0.0)

    # Runs RP to its end with the run's Picard settings, recording after each step the plasma
    # current, the induced field and the coil currents.
    function run_record(RP)
        rec = (I = Float64[], E = Matrix{Float64}[], Ic = Vector{Float64}[])
        function record(rp)
            push!(rec.I, sum(rp.plasma.Jϕ) * rp.G.dR * rp.G.dZ)
            push!(rec.E, copy(rp.fields.Eϕ_self))
            push!(rec.Ic, copy(get_all_currents(rp.coil_system)))
            return nothing
        end
        redirect_stderr(devnull) do
            redirect_stdout(() -> run_simulation!(RP; callback_after_step = record), devnull)
        end
        return rec
    end

    # Each coil's error over the run against its own peak current (1 µA at least), so that a
    # strong coil does not hide a weak one.
    function coil_error(ra, rb)
        isempty(rb.Ic[1]) && return 0.0
        Ia, Ib = reduce(hcat, ra.Ic), reduce(hcat, rb.Ic)   # coil × step
        return maximum(maximum(abs, Ia[c, :] .- Ib[c, :]) / max(maximum(abs, Ib[c, :]), 1.0e-6) for c in axes(Ib, 1))
    end

    # The default run against the direct one, step by step: the plasma current, the induced
    # field over the grid, each coil current, and u∥ and ψ_self at the end.
    function compare_with_converged(make)
        a, b = make(), make()
        b.flags.ampere_picard = PicardSettings{Float64}(; REFERENCE_PICARD...)
        ra, rb = run_record(a), run_record(b)
        rel(x, y) = maximum(abs, x .- y) / maximum(abs, y)
        return (;
            a, b, unconverged = (a.diagnostics.ampere_picard.nunconverged, b.diagnostics.ampere_picard.nunconverged),
            I = maximum(abs(ra.I[k] - rb.I[k]) / abs(rb.I[k]) for k in eachindex(rb.I)),
            E = maximum(rel(ra.E[k], rb.E[k]) for k in eachindex(rb.E)),
            Ic = coil_error(ra, rb),
            u = rel(a.plasma.ue_para, b.plasma.ue_para), ψ = rel(a.fields.ψ_self, b.fields.ψ_self),
        )
    end
end

@testitem "Coupled solve: a column that fills a tight box takes the converged step" setup = [BoundaryLimitedColumn] begin
    # One mode of the outer iteration has eigenvalue near −4 here, so the relaxed iteration with
    # w = 1/2 diverges.
    c = compare_with_converged(tight_column)
    @test c.unconverged == (0, 0)
    @test c.I <= 1.0e-3
    @test c.E <= 2.0e-3
    @test c.u <= 2.0e-3 && c.ψ <= 2.0e-3
end

@testitem "Coupled solve: a dense column inside a filament shell takes the converged step" setup = [BoundaryLimitedColumn] begin
    # The shell gives modes with eigenvalues up to 0.85: slow for any relaxation weight.
    c = compare_with_converged(shell_column)
    @test c.unconverged == (0, 0)
    @test c.I <= 1.0e-3
    @test c.E <= 2.0e-3
    @test c.Ic <= 2.0e-3
    @test c.u <= 2.0e-3 && c.ψ <= 2.0e-3
end

@testitem "Coupled solve: advection and coils of very different inductance" setup = [BoundaryLimitedColumn] begin
    c = compare_with_converged(mixed_column)
    @test c.unconverged == (0, 0)
    @test c.I <= 1.0e-3
    @test c.E <= 2.0e-3
    @test c.Ic <= 2.0e-3
    @test c.u <= 2.0e-3 && c.ψ <= 2.0e-3
end

@testitem "Coupled solve: the coils remember the flux of the accepted block-stage current" setup = [BoundaryLimitedColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!, plasma_flux_at_coils
    # Right after the solve, before the step goes on, Jϕ is the current the solve accepted; the
    # flux each coil remembers must be that current's, whether the solve converged or was cut
    # short.
    for picard in ((;), (tolerance = 1.0e-300, max_iter = 2))
        RP = shell_column()
        RP.plasma.ne[RP.G.nodes.on_out_wall_nids] .= 0.0
        RP.plasma.ni[RP.G.nodes.on_out_wall_nids] .= 0.0
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
        redirect_stderr(() -> solve_combined_momentum_Ampere_equations_with_coils!(RP; picard...), devnull)
        Φ = plasma_flux_at_coils(RP.coil_system, RP.G, RP.plasma.Jϕ)
        @test RP.coil_system.coils.ψ_pla ≈ Φ rtol = 1.0e-12
    end
end

@testitem "Coupled solve: the fixed point does not depend on the mixing, and the direct solve finds it" setup = [PicardColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # The easy column with a loop, one step solved tightly with the relaxed iteration (memory 0),
    # with Anderson mixing (memory 8), and directly.
    function solved(method)
        RP = picard_column(; t_end = 20.0e-6)
        add_coil!(
            RP.coil_system, Coil{Float64}(;
                location = (r = 1.2, z = 0.8), area = π * 0.05^2, resistance = 1.0e-3,
                self_inductance = 1.2e-6, is_powered = false, is_controllable = false, name = "loop",
            )
        )
        initialize_coil_system!(RP)
        redirect_stdout(() -> run_simulation!(RP), devnull)
        prepare_timestep!(RP)
        solve_combined_momentum_Ampere_equations_with_coils!(
            RP; method, tolerance = 1.0e-13, max_iter = 500, E_floor = 0.0, I_floor = 0.0,
        )
        return RP
    end
    a, b, c = solved(AndersonOuterSolve(; memory = 0)), solved(AndersonOuterSolve()), solved(DirectOuterSolve())
    @test c.diagnostics.ampere_picard.last_niter == 2   # the first evaluation, then Newton's step
    agree(x, y) = maximum(abs, x .- y) <= 1.0e-10 * maximum(abs, y)
    for other in (b, c)
        @test agree(a.plasma.ue_para, other.plasma.ue_para)
        @test agree(a.fields.ψ_self, other.fields.ψ_self)
        @test agree(a.coil_system.coils.current, other.coil_system.coils.current)
        @test agree(a.coil_system.coils.ψ_pla, other.coil_system.coils.ψ_pla)
    end
end

@testitem "Coupled solve: the direct solve takes the step at once where the iteration struggles" setup = [BoundaryLimitedColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # The first step of each hard column: a mode near −4 (tight box), modes near 1 (shell), coil
    # inductances over four decades (mixed). Solved directly to 1e-12 of the step, with no
    # floors: the first evaluation, then Newton's step, which already meets that.
    for make in (tight_column, shell_column, mixed_column)
        RP = make()
        RP.plasma.ne[RP.G.nodes.on_out_wall_nids] .= 0.0
        RP.plasma.ni[RP.G.nodes.on_out_wall_nids] .= 0.0
        initialize_coupled_fields!(RP)
        prepare_timestep!(RP)
        solve_combined_momentum_Ampere_equations_with_coils!(
            RP; method = DirectOuterSolve(), tolerance = 1.0e-12, E_floor = 0.0, I_floor = 0.0,
        )
        stats = RP.diagnostics.ampere_picard
        @test stats.nunconverged == 0
        @test stats.last_niter == 2
    end
end

@testitem "Coupled solve: a residual that is not finite never converges" begin
    using RAPID2D: coupled_residual_converged
    R = fill(1.5, 2, 2)
    R_b = fill(1.5, 2)
    check(f_ψb, ψ_new; f_I = Float64[], coil_field = Float64[], I_new = Float64[], I_old = Float64[]) =
        coupled_residual_converged(
        f_ψb, f_I, R_b, coil_field, ψ_new, zeros(2, 2), R, I_new, I_old, 1.0e-6; tol = 1.0e-3, E_floor = 1.0e-6, I_floor = 1.0e-6,
    )[1]
    @test !check([Inf, 0.0], fill(Inf, 2, 2))   # an infinite residual against an infinite step
    @test !check([NaN, 0.0], ones(2, 2))
    @test !check([0.0, 0.0], fill(NaN, 2, 2))
    @test !check([0.0, 0.0], ones(2, 2); f_I = [Inf], coil_field = [0.0], I_new = [Inf], I_old = [0.0])
end

@testitem "Coupled solve: each part of the stopping test can hold a solve back" begin
    using RAPID2D: coupled_residual_converged
    # A step that induces 1 V/m: the boundary residual is well inside the tolerance, and
    # either one coil's current or the in-grid source is not.
    R, R_b, dt = fill(1.5, 2, 2), fill(1.5, 2), 1.0e-6
    ψ_new = fill(1.5 * dt, 2, 2)                        # E_step = 1 V/m
    f_ψb = fill(1.0e-9 * 1.5 * dt, 2)                     # 1e-9 V/m
    conv(f_I, coil_field, I_new, I_old) = coupled_residual_converged(
        f_ψb, f_I, R_b, coil_field, ψ_new, zeros(2, 2), R, I_new, I_old, dt; tol = 1.0e-3, E_floor = 0.0, I_floor = 0.0,
    )[1]
    # two coils outside the grid with the same residual: settled for the one that changes by
    # 100 A over the step, not for the one that changes by 1 mA
    @test conv([1.0e-5, 1.0e-5], [0.0, 0.0], [100.0, 1.0e-3], [0.0, 0.0]) == false
    @test conv([1.0e-5, 1.0e-7], [0.0, 0.0], [100.0, 1.0e-3], [0.0, 0.0]) == true
    # a coil inside the grid whose current has settled but whose source still moves the field
    @test conv([1.0e-3], [10.0], [100.0], [0.0]) == false   # 1e-3 A × 10 V/m/A = 1e-2 V/m > 1e-3 V/m
    @test conv([1.0e-5], [10.0], [100.0], [0.0]) == true
end

@testitem "Ampère Picard: the run's floors reach the coupled solve" setup = [PicardColumn] begin
    # Floors so large that the first block solve of every step settles it.
    RP = picard_column(; t_end = 20.0e-6)
    RP.flags.ampere_picard = PicardSettings{Float64}(; tolerance = 0.0, E_floor = 1.0e30, I_floor = 1.0e30)
    redirect_stdout(() -> run_simulation!(RP), devnull)
    stats = RP.diagnostics.ampere_picard
    @test stats.nsolve == 4
    @test stats.niter == stats.nsolve
    @test stats.nunconverged == 0
end

@testitem "Coupled solve: a solve that does not converge keeps its best evaluation" setup = [BoundaryLimitedColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # The first step of the tight column with a loop outside the grid, relaxed (m = 0) with a
    # weight of 4, or of 10⁶: every iterate after the first is worse. Cut at max_iter = 3, or
    # stopped by its exhausted restarts, the solve accepts its first evaluation, as the solve
    # cut at max_iter = 1 does: u∥, ψ, the coil currents, their memory and the residual.
    function first_step(; relaxation_w = 0.5, picard...)
        RP = tight_column()
        add_coil!(
            RP.coil_system, Coil{Float64}(;
                location = (r = 2.4, z = 0.0), area = π * 0.05^2, resistance = 1.0e-3,
                self_inductance = 1.3e-5, is_powered = false, is_controllable = false, name = "loop",
            )
        )
        initialize_coil_system!(RP)
        prepare_timestep!(RP)
        redirect_stderr(devnull) do
            solve_combined_momentum_Ampere_equations_with_coils!(
                RP; method = AndersonOuterSolve(; memory = 0, relaxation_w), picard...,
            )
        end
        return RP
    end
    function same_as_first(RP, ref)
        a, b = RP.diagnostics.ampere_picard, ref.diagnostics.ampere_picard
        return a.nunconverged == 1 && RP.plasma.ue_para == ref.plasma.ue_para &&
            RP.fields.ψ_self == ref.fields.ψ_self && RP.fields.Eϕ_self == ref.fields.Eϕ_self &&
            RP.coil_system.coils.current == ref.coil_system.coils.current &&
            RP.coil_system.coils.ψ_pla == ref.coil_system.coils.ψ_pla &&
            (a.last_E_residual, a.last_I_residual) == (b.last_E_residual, b.last_I_residual)
    end
    ref = first_step(; max_iter = 1)
    cut = first_step(; relaxation_w = 4.0, max_iter = 3)
    exhausted = first_step(; relaxation_w = 1.0e6, max_iter = 100)
    @test cut.diagnostics.ampere_picard.last_niter == 3
    @test 3 < exhausted.diagnostics.ampere_picard.last_niter < 100
    @test same_as_first(cut, ref)
    @test same_as_first(exhausted, ref)
end

@testitem "Coupled solve: an evaluation whose induced field overflows is never accepted" begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # A vacuum in a purely poloidal field (bϕ = 0, so u∥ does not see ψ), entering the step with
    # ψ = 1e305 inside and 0 on the boundary. The block solve's ψ is finite, but the field it
    # induces over the step, −Δψ/(R Δt), is not. Such an evaluation is not accepted, not even as
    # the best one: with nothing else to fall back on, the solve throws.
    config = SimulationConfig{Float64}(;
        device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = 0.1, Eϕ = 0.0),
        NR = 10, NZ = 12, R0B0 = 0.0, prefilled_gas_pressure = 0.0, dt = 1.0e-6, t_end_s = 1.0e-6,
        snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 1.0e-6, Output_path = mktempdir(; cleanup = false),
    )
    RP = RAPID{Float64}(config)
    RP.flags = SimulationFlags{Float64}(;
        Ampere = true, Ampere_Itor_threshold = 0.0, E_para_self_EM = true, ud_evolve = true,
        Coulomb_Collision = true, Atomic_Collision = false, src = false, convec = false, diffu = false,
        Te_evolve = false, Ti_evolve = false, Gas_evolve = false, update_ni_independently = false,
        Include_ud_convec_term = false, Include_ud_diffu_term = false, Include_ud_pressure_term = false,
        E_para_self_ES = false, mean_ExB = false, turb_ExB_mixing = false, FLF_nstep = 100_000,
    )
    initialize!(RP)
    fill!(RP.plasma.Te_eV, 1.0)
    RAPID2D.update_transport_quantities!(RP)
    RP.fields.ψ_self[setdiff(eachindex(RP.fields.ψ_self), RP.G.BDY_idx)] .= 1.0e305
    prepare_timestep!(RP)
    @test all(iszero, RP.fields.bϕ)
    @test_throws ErrorException solve_combined_momentum_Ampere_equations_with_coils!(RP; max_iter = 1)
end

@testitem "Coupled solve: a solve that throws leaves the state as it found it" setup = [PicardColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # The column a few steps in, with a loop added since (its plasma flux unset), and a drive
    # that is not finite in one cell: the first evaluation is not finite, so the solve throws.
    # What the solve writes on acceptance must stay as it was: the induced field the predictor
    # extrapolates from, the friction and the loop's memory.
    RP = picard_column(; t_end = 20.0e-6)
    redirect_stdout(() -> run_simulation!(RP), devnull)
    add_coil!(
        RP.coil_system, Coil{Float64}(;
            location = (r = 1.2, z = 0.8), area = π * 0.05^2, resistance = 1.0e-3,
            self_inductance = 1.2e-6, is_powered = false, is_controllable = false, name = "loop",
        )
    )
    initialize_coil_system!(RP)
    prepare_timestep!(RP)
    RP.fields.E_para_ext[argmax(RP.plasma.ne)] = NaN
    F, pla, csys = RP.fields, RP.plasma, RP.coil_system
    state() = (F.ψ_self, F.Eϕ_self, F.Eϕ_self_prev, pla.ue_para, pla.Rue_ei, csys.coils.current, csys.coils.ψ_pla, csys.time_s)
    before = deepcopy(state())
    @test !isequal(F.Eϕ_self, F.Eϕ_self_prev) && isnan(csys.coils.ψ_pla[1])
    @test_throws ErrorException solve_combined_momentum_Ampere_equations_with_coils!(RP)
    @test all(isequal.(state(), before))
end

@testitem "Ampère Picard: the run's coil floor reaches the coupled solve" setup = [PicardColumn] begin
    # The column with a loop, the field part of the stopping test always met (E_floor = 1e30):
    # whether a solve stops at its first evaluation then depends only on the coils' floor.
    function stats(I_floor)
        RP = picard_column(; t_end = 20.0e-6)
        add_coil!(
            RP.coil_system, Coil{Float64}(;
                location = (r = 1.2, z = 0.8), area = π * 0.05^2, resistance = 1.0e-3,
                self_inductance = 1.2e-6, is_powered = false, is_controllable = false, name = "loop",
            )
        )
        initialize_coil_system!(RP)
        RP.flags.ampere_picard = PicardSettings{Float64}(; tolerance = 0.0, max_iter = 3, E_floor = 1.0e30, I_floor)
        redirect_stderr(devnull) do
            redirect_stdout(() -> run_simulation!(RP), devnull)
        end
        return RP.diagnostics.ampere_picard
    end
    loose, tight = stats(1.0e30), stats(0.0)
    @test loose.niter == loose.nsolve == 4 && loose.nunconverged == 0
    @test tight.nunconverged == tight.nsolve == 4
end

@testitem "Ampère Picard: the settings are checked where they are written" begin
    s = PicardSettings{Float64}()
    @test NamedTuple(s) == (
        method = AndersonOuterSolve(; memory = 8, relaxation_w = 0.5), tolerance = 1.0e-3, max_iter = 20,
        E_floor = 1.0e-6, I_floor = 1.0e-6,
    )
    @test PicardSettings{Float32}().tolerance isa Float32
    s.max_iter = 40
    s.method = DirectOuterSolve()
    @test (s.max_iter, s.method) == (40, DirectOuterSolve())
    # each out of range on construction and on assignment; a method is a policy, not a name
    for (name, bad) in (
            (:tolerance, -1.0), (:tolerance, NaN), (:max_iter, 0), (:E_floor, -1.0e-6), (:I_floor, Inf),
            (:method, :direct),
        )
        @test_throws ArgumentError PicardSettings{Float64}(; (name => bad,)...)
        @test_throws ArgumentError setproperty!(PicardSettings{Float64}(), name, bad)
    end
    # the policy checks its own parameters; a weight above 1 over-relaxes
    @test_throws ArgumentError AndersonOuterSolve(; memory = -1)
    @test_throws ArgumentError AndersonOuterSolve(; relaxation_w = 0.0)
    @test_throws ArgumentError AndersonOuterSolve(; relaxation_w = NaN)
    @test_throws ArgumentError AndersonOuterSolve(; relaxation_w = big"1e400")    # Inf as a Float64
    @test_throws ArgumentError AndersonOuterSolve(; relaxation_w = big"1e-400")   # 0.0 as a Float64
    @test AndersonOuterSolve(; relaxation_w = 1.5).relaxation_w == 1.5
end

@testitem "Coupled solve: the u∥–ψ block keeps one sparsity pattern" setup = [PicardColumn] begin
    using RAPID2D: solve_combined_momentum_Ampere_equations_with_coils!
    # With u∥ advection on, the upwind side each face takes follows the flow. Turning the flow
    # around between solves changes only the block's values: one symbolic analysis for all.
    RP = picard_column()
    RP.flags.Include_ud_convec_term = true
    for (uR, uZ) in ((1.0e3, 0.0), (-1.0e3, 0.0), (0.0, 5.0e2), (-1.0e3, -5.0e2))
        prepare_timestep!(RP)
        fill!(RP.plasma.ueR, uR)
        fill!(RP.plasma.ueZ, uZ)
        RAPID2D.cache_electron_operators!(RP)
        solve_combined_momentum_Ampere_equations_with_coils!(RP)
    end
    @test RP.operators.uψ_solver.nfactor == 4
    @test RP.operators.uψ_solver.nsymbolic == 1
end

@testitem "Coupled solve: the fixed-pattern block is the block matrix sparse algebra assembles" setup = [PicardColumn] begin
    using RAPID2D: CoupledBlock, update_coupled_block!, uψ_coupling_terms
    using RAPID2D: set_identity!, add_diagonal!, add_scaled!, ue_Te_operators
    using RAPID2D.SparseArrays
    RP = picard_column()
    RP.flags.Include_ud_convec_term = true
    fill!(RP.plasma.ueR, 1.0e3)
    fill!(RP.plasma.ueZ, -4.0e2)
    RAPID2D.cache_electron_operators!(RP)
    OP, dt = RP.operators, RP.dt
    ν = vec(RP.plasma.ν_en_mom_tot + RP.plasma.ν_en_iz_tot + RP.plasma.ν_ei_eff)
    A_adv = ue_Te_operators(RP).A_adv
    # 𝟙 + Δt ν + Δt A_adv and the coupling diagonals, put together by sparse algebra
    c_uψ, c_ψu = uψ_coupling_terms(RP)
    A_u_ref = OP.II + spdiagm(dt .* ν) + dt * sparse(A_adv)
    ref = [A_u_ref spdiagm(c_uψ); spdiagm(c_ψu) OP.ΔGS.matrix]
    # the same on the wall pattern and a block built once
    A_u = similar(OP.A_adv_e)
    set_identity!(A_u)
    add_diagonal!(A_u, ν; scale = dt)
    add_scaled!(A_u, dt, A_adv)
    B = CoupledBlock(A_u, OP.ΔGS)
    new = update_coupled_block!(B, A_u, OP.ΔGS, c_uψ, c_ψu)
    @test new == ref
    @test nnz(new) > nnz(ref)   # the upwind sides a flow does not use are stored zeros
end
