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
