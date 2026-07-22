# Basic inductance regression scenario: a loop-voltage-driven plasma filament checked
# against two analytical L/R references — constant inductance, and the time-varying L(t)
# reported by the 0D diagnostics.
#
# ONE @testitem: all 5 assertions consume the same ~8.6 s simulation, and a snippet body
# is re-evaluated per testitem, so splitting would re-run it per group.

@testsnippet InductanceAnalysis begin
    using RAPID2D.Statistics
    using RAPID2D.FastInterpolations
    using Printf
    using Plots
    using Dates

    # Compare the simulated toroidal current against analytical L/R solutions.
    # `major_R` / `minor_r` [m] are required kwargs: the inductance estimate must stay
    # tied to the filament the scenario actually created.
    function analyze_inductance(
            RP::RAPID; major_R::Real, minor_r::Real,
            verbose::Bool = false, visualize::Bool = false,
            outdir::AbstractString = mktempdir(; cleanup = false)
        )
        if verbose
            println("\n" * "="^60)
            println("INDUCTANCE ANALYSIS")
            println("="^60)
        end

        # Extract time series data
        times = RP.diagnostics.snaps0D.time_s
        I_tor = RP.diagnostics.snaps0D.I_tor

        RAPID2D.@unpack ee, me = RP.config.constants
        # Resistance by Coulomb collision
        ue_sat_by_Evac = @. -ee * RP.fields.Eϕ_ext / (me * RP.plasma.ν_ei_eff)
        ue_sat_by_Evac[.!isfinite.(ue_sat_by_Evac)] .= 0.0  # Avoid NaN

        # Estimate circuit parameters
        in_wall_nids = RP.G.nodes.in_wall_nids
        LV_estimate = mean(RP.fields.LV_ext[in_wall_nids])
        R_estimate = LV_estimate ./ sum(-ee * RP.plasma.ne[in_wall_nids] .* ue_sat_by_Evac[in_wall_nids] * RP.G.dR * RP.G.dZ)

        # Inductance of a circular loop of major radius `major_R`, minor radius `minor_r`
        Y = 1.0  # Internal inductance factor
        μ0 = 4π * 1.0e-7  # H/m
        L_estimate = μ0 * major_R * (log(8 * major_R / minor_r) - 2 + 0.25 * Y)

        # Estimate mutual inductance (geometric calculation)
        snap0D_time_s = RP.diagnostics.snaps0D.time_s
        snap0D_time_s = range(snap0D_time_s[1], stop = snap0D_time_s[end], length = length(snap0D_time_s))
        L_values = RP.diagnostics.snaps0D.self_inductance_plasma
        L_values[1] = L_values[2] # Avoid zero at t=0
        itp_L_self_plasma = cubic_interp(snap0D_time_s, L_values)

        # L/R time constant
        tau_LR = L_estimate / R_estimate

        if verbose
            println("Circuit Parameter Estimates:")
            println(@sprintf("  Loop voltage: %.3f V", LV_estimate))
            println(@sprintf("  Final current: %.3f A", I_tor[end]))
            println(@sprintf("  Resistance: %.6f Ω", R_estimate))
            println(@sprintf("  Inductance: %.6f μH", L_estimate * 1.0e6))
            println(@sprintf("  L/R time: %.1f μs", tau_LR * 1.0e6))
        end

        # Analytical solution (constant L approximation)
        I_sat_analytical = LV_estimate / R_estimate
        I_analytical = @. I_sat_analytical * (1 - exp(-times / tau_LR))

        # More accurate analytical solution with time-varying inductance
        I_analytical_timevar = calculate_time_varying_inductance_solution(times, LV_estimate, R_estimate, itp_L_self_plasma)

        # Create plots
        if visualize
            create_inductance_plots(RP, times, I_tor, I_analytical, I_analytical_timevar; outdir)
        end

        # Calculate error metrics and return for testing
        if length(I_tor) == length(I_analytical) && length(I_tor) == length(I_analytical_timevar)
            # Error vs constant L analytical solution
            relative_error = abs.(I_tor - I_analytical) ./ (abs.(I_analytical))
            relative_error[.!isfinite.(relative_error)] .= 0.0  # Avoid NaN & Inf
            mean_error = mean(relative_error)
            max_error = maximum(relative_error)

            # Error vs time-varying L analytical solution
            relative_error_timevar = abs.(I_tor - I_analytical_timevar) ./ (abs.(I_analytical_timevar))
            relative_error_timevar[.!isfinite.(relative_error_timevar)] .= 0.0  # Avoid NaN & Inf
            mean_error_timevar = mean(relative_error_timevar)
            max_error_timevar = maximum(relative_error_timevar)

            if verbose
                println("\nAccuracy Assessment:")
                println("  === vs Constant L Analytical ===")
                println(@sprintf("  Mean relative error: %.2f%%", 100 * mean_error))
                println(@sprintf("  Max relative error: %.2f%%", 100 * max_error))

                println("  === vs Time-varying L Analytical ===")
                println(@sprintf("  Mean relative error: %.2f%%", 100 * mean_error_timevar))
                println(@sprintf("  Max relative error: %.2f%%", 100 * max_error_timevar))
            end

            return (
                mean_error = mean_error, max_error = max_error,
                mean_error_timevar = mean_error_timevar, max_error_timevar = max_error_timevar,
                times = times, I_tor = I_tor, I_analytical = I_analytical, I_analytical_timevar = I_analytical_timevar,
            )
        else
            return nothing
        end
    end

    # Create visualization plots for inductance test results.
    # `outdir` MUST be unique per run: the filename timestamp is only second-resolution,
    # so two runs sharing a directory can overwrite each other.
    function create_inductance_plots(
            RP, times, I_sim, I_analytical, I_analytical_timevar;
            outdir::AbstractString = mktempdir(; cleanup = false)
        )
        # Current evolution plot
        p1 = plot(
            times * 1.0e3, I_sim,
            label = "Simulation",
            linewidth = 2,
            xlabel = "Time (ms)",
            ylabel = "Toroidal Current (A)",
            title = "L/R Circuit Response"
        )

        plot!(
            p1, times * 1.0e3, I_analytical,
            label = "Analytical (const L)",
            linestyle = :dash,
            linewidth = 2
        )

        plot!(
            p1, times * 1.0e3, I_analytical_timevar,
            label = "Analytical (time-var L)",
            linestyle = :dot,
            linewidth = 2,
            color = :green
        )

        # Error plots
        if length(I_sim) == length(I_analytical) && length(I_sim) == length(I_analytical_timevar)
            error_const_L = abs.(I_sim - I_analytical) ./ (abs.(I_analytical)) * 100
            error_timevar_L = abs.(I_sim - I_analytical_timevar) ./ (abs.(I_analytical_timevar)) * 100

            p2 = plot(
                times * 1.0e3, error_const_L,
                label = "vs Const L",
                linewidth = 2,
                xlabel = "Time (ms)",
                ylabel = "Error (%)",
                title = "Simulation Errors"
            )

            plot!(
                p2, times * 1.0e3, error_timevar_L,
                label = "vs Time-var L",
                linestyle = :dash,
                linewidth = 2,
                color = :green
            )
        else
            p2 = plot(title = "Error analysis unavailable")
        end

        # Inductance evolution plot
        snap0D_times = RP.diagnostics.snaps0D.time_s
        L_values = RP.diagnostics.snaps0D.self_inductance_plasma

        p3 = plot(
            snap0D_times * 1.0e3, L_values * 1.0e6,
            label = "L_self_plasma(t)",
            linewidth = 2,
            xlabel = "Time (ms)",
            ylabel = "Self-Inductance (μH)",
            title = "Time-Varying Plasma Inductance"
        )

        # Combined plot
        plot_combined = plot(p1, p2, p3, layout = (3, 1), size = (800, 900))

        # Save plot into the caller-provided unique directory
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
        full_path = joinpath(outdir, "inductance_test_$(timestamp).png")
        savefig(plot_combined, full_path)
        @info "Inductance test plot saved" path = full_path

        # Create animation if possible
        anim_path = joinpath(outdir, "inductance_test_snaps2D_$(timestamp).mp4")
        try
            animate_snaps2D(
                RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                [:ne, :ue_para, :Jϕ, :E_para_tot];
                wall = RP.fitted_wall,
                filename = anim_path
            )
            @info "Inductance test animation saved" path = anim_path
        catch e
            @warn "Inductance test animation creation failed" exception = e
        end

        return plot_combined
    end

    # Analytical solution for an RL circuit with time-varying inductance L(t).
    # Solves L(t)*dI/dt + dL/dt*I + R*I = V by 4th-order Runge-Kutta.
    function calculate_time_varying_inductance_solution(times, V, R, itp_L; I0 = 0.0)
        dt = length(times) > 1 ? times[2] - times[1] : 1.0e-6
        I_solution = zeros(length(times))
        I_solution[1] = I0

        for i in 2:length(times)
            t_prev = times[i - 1]
            t_curr = times[i]
            I_prev = I_solution[i - 1]

            # Define the ODE: dI/dt = (V - R*I - dL/dt*I) / L(t)
            function ode_func(t, I)
                L_t = itp_L(t)

                # Calculate dL/dt numerically
                dt_small = 1.0e-8
                if t + dt_small <= times[end]
                    dL_dt = (itp_L(t + dt_small) - L_t) / dt_small
                else
                    dL_dt = (L_t - itp_L(t - dt_small)) / dt_small
                end

                # dI/dt = (V - R*I - dL/dt*I) / L(t)
                if L_t > 0
                    return (V - R * I - dL_dt * I) / L_t
                else
                    return 0.0
                end
            end

            # 4th-order Runge-Kutta integration
            k1 = ode_func(t_prev, I_prev)
            k2 = ode_func(t_prev + dt / 2, I_prev + dt * k1 / 2)
            k3 = ode_func(t_prev + dt / 2, I_prev + dt * k2 / 2)
            k4 = ode_func(t_curr, I_prev + dt * k3)

            I_solution[i] = I_prev + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        end

        return I_solution
    end
end

@testitem "Basic Inductance" tags = [:regression] setup = [RegressionCommon, InductanceAnalysis] begin
    # Declared here as well as in the snippets: a snippet becomes a module under the
    # ReTestItems path, and `using` only re-exports names a module OWNS, so imports made
    # inside a snippet are invisible to this body.
    using RAPID2D.Statistics
    using Printf

    # A single plasma filament driven by a toroidal loop voltage. Everything that is not
    # L/R circuit physics is switched off, so the current response can be compared
    # against an analytical L/R solution.

    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

    # Artifacts go to a fresh per-run directory; cleanup=false so they outlive the
    # process and stay inspectable. The path is reported by @info when plots are saved.
    outdir = visualize ? mktempdir(; cleanup = false) : tempname()

    config = regression_config(;
        NR = 30,
        NZ = 50,
        prefilled_gas_pressure = 0.0,   # vacuum: no neutrals, hence no atomic physics
        R0B0 = 3.0,                     # T⋅m, pure toroidal field
        dt = 25.0e-6,
        t_end_s = 20.0e-3,                # ≫ L/R time, so the current reaches saturation
        snap0D_Δt_s = 50.0e-6,            # every 2 steps -> 401 points to compare against
        snap2D_Δt_s = 500.0e-6,
    )

    RP = RAPID{Float64}(config)

    # ── The physics under test ───────────────────────────────────────────────
    # Ampère's law evolves the current, ud_evolve accelerates the electrons along B, and
    # E_para_self_EM (set below) supplies the inductive back-EMF that opposes it. Those
    # three together ARE the L/R circuit. Everything else is off so that nothing else can
    # move current or particles around.
    RP.flags = SimulationFlags{Float64}(
        Ampere = true,                        # <- magnetic field update: the physics under test
        ud_evolve = true,                     # <- parallel electron acceleration drives I_tor
        negative_n_correction = true,

        # Everything below is deliberately disabled: with these on, the plasma would
        # no longer be a fixed lumped circuit element.
        Atomic_Collision = false,             # no ionization/losses -> density is frozen
        Te_evolve = false,                    # isothermal
        Ti_evolve = false,
        src = false,                          # no particle sources
        convec = false,                       # no transport: the filament must not move
        diffu = false,                        #   or spread, or L(t) would drift
        Gas_evolve = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,       # momentum equation reduced to dU/dt = -eE/m - νU
        Include_ud_diffu_term = false,
        Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false,               # electrostatic self-field is NOT the effect studied
    )

    RP.flags.Coulomb_Collision = true         # the only resistivity source -> the "R" in L/R
    RP.flags.E_para_self_EM = true            # inductive back-EMF: the effect being measured
    RP.flags.Ampere_Itor_threshold = 0.0      # apply Ampère from the very first amp
    RP.flags.FLF_nstep = 100
    RP.flags.Implicit = true
    RP.flags.Damp_Transp_outWall = true
    RP.flags.Global_JxB_Force = false         # not needed for basic inductance test

    initialize!(RP)

    # Pure toroidal field plus a loop voltage. E0 > 0 is what drives the circuit.
    setup_toroidal_field!(RP; E0 = 0.3, verbose)

    # ── Initial plasma: one uniform top-hat filament ─────────────────────────
    # Exactly zero outside the minor radius, uniform Te/Ti, at rest. The same geometry
    # feeds the analytical inductance estimate below, so it is named once here.
    filament_R = 1.5    # major radius of the current filament [m]
    filament_a = 0.3    # minor radius [m]
    filament_n0 = 1.0e16   # uniform density inside the filament [m⁻³]

    ini_n = tophat_blob(RP.G; cenR = filament_R, radius = filament_a, n0 = filament_n0)
    RP.plasma.ne .= ini_n
    RP.plasma.ni .= ini_n

    fill!(RP.plasma.Te_eV, 10.0)      # eV
    fill!(RP.plasma.Ti_eV, 0.03)      # eV (room temperature)
    fill!(RP.plasma.ue_para, 0.0)     # start from zero current
    fill!(RP.plasma.ui_para, 0.0)

    run_simulation!(RP)

    # Compare I_tor(t) against the constant-L and time-varying-L analytical solutions.
    results = analyze_inductance(
        RP; major_R = filament_R, minor_r = filament_a,
        verbose, visualize, outdir
    )

    @test results !== nothing  # Should return valid results

    if results !== nothing
        @test results.mean_error < 0.03  # Mean error should be less than 3%
        @test results.max_error < 0.05   # Max error should be less than 5%
        @test results.I_tor[end] > 0     # Final current should be positive
        @test length(results.times) > 1  # Should have multiple time points
    end
end
