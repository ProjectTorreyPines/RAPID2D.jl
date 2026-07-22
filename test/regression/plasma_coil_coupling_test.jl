# Plasma-coil inductive coupling regression scenario.
#
# A stationary plasma column and a single external coil form two magnetically coupled
# LR circuits. The plasma loop is driven by the loop voltage, the coil carries no
# applied voltage at all, and the only thing linking them is the mutual inductance —
# so the coil current is a direct measurement of the inductive coupling. The simulated
# pair (I_coil(t), I_plasma(t)) is compared against two analytical solutions:
#   * constant M and L₂ — closed form via eigen-decomposition of -L⁻¹R;
#   * time-varying M(t) and L₂(t) taken from the diagnostics — RK4 integration.
# The assertions use the second (tighter) comparison.
#
# ONE @testitem: all 8 assertions consume the same ~65 s simulation, and a snippet body
# is re-evaluated per testitem, so splitting would re-run it per group.

@testsnippet CouplingAnalysis begin
    using RAPID2D.Statistics
    using RAPID2D.LinearAlgebra
    using RAPID2D.FastInterpolations
    using Printf
    using Plots
    using Dates

    # Analytical solution for two coupled LR circuits
    #
    # System: L₁(dI₁/dt) + M(dI₂/dt) + R₁I₁ = V₁
    #         L₂(dI₂/dt) + M(dI₁/dt) + R₂I₂ = V₂
    #
    # Where: Circuit 1 = External coil, Circuit 2 = Plasma loop
    function calculate_analytical_solution(times, L1, L2, M, R1, R2, V1, V2; I1_0 = 0.0, I2_0 = 0.0)
        # System matrices
        L_matrix = [L1 M; M L2]
        R_matrix = [R1 0.0; 0.0 R2]
        V_vector = [V1; V2]

        # System matrix A = -L⁻¹R
        A_sys = -L_matrix \ R_matrix

        # Eigenvalues and eigenvectors
        eigenvals, eigenvecs = eigen(A_sys)
        λ1, λ2 = eigenvals[1], eigenvals[2]
        v1, v2 = eigenvecs[:, 1], eigenvecs[:, 2]

        # Steady-state solution
        L_matrix = Array(L_matrix)  # Ensure Array type
        R_matrix = Array(R_matrix)
        I_steady = R_matrix \ V_vector

        # Initial condition: I(0) = [I1_0; I2_0]
        # Solution: I(t) = I_steady + c1*v1*exp(λ1*t) + c2*v2*exp(λ2*t)
        # From I(0): I_steady + c1*v1 + c2*v2 = [I1_0; I2_0]
        coeff_matrix = [v1 v2]
        coeffs = coeff_matrix \ ([I1_0; I2_0] - I_steady)
        c1, c2 = coeffs[1], coeffs[2]

        # Calculate solution at all time points
        I1_analytical = zeros(length(times))
        I2_analytical = zeros(length(times))

        for (i, t) in enumerate(times)
            I_t = I_steady + c1 * v1 * exp(λ1 * t) + c2 * v2 * exp(λ2 * t)
            I1_analytical[i] = I_t[1]  # Coil current
            I2_analytical[i] = I_t[2]  # Plasma current
        end

        return I1_analytical, I2_analytical, λ1, λ2, I_steady
    end

    # Semi-analytical solution for two coupled LR circuits with time-varying mutual
    # inductance M(t) and plasma inductance L₂(t)
    #
    # System (from Faraday's law):
    # V₁ = L₁(dI₁/dt) + M(t)(dI₂/dt) + (dM/dt)I₂ + R₁I₁
    # V₂ = L₂(t)(dI₂/dt) + (dL₂/dt)I₂ + M(t)(dI₁/dt) + (dM/dt)I₁ + R₂I₂
    #
    # Where: Circuit 1 = External coil (constant L₁), Circuit 2 = Plasma loop
    #        (time-varying L₂(t)); itp_M returns M(t); itp_L_plasma returns L₂(t).
    #
    # Uses 4th-order Runge-Kutta with fine time steps for accuracy, but only stores
    # results at the requested time points in the 'times' array.
    function calculate_analytical_solution_time_varying_M_and_L(times, L1, itp_L_plasma, itp_M, R1, R2, V1, V2; I1_0 = 0.0, I2_0 = 0.0, dt = nothing)
        # Use small time step for accuracy (RP.dt or smaller)
        if dt === nothing
            dt_requested = length(times) > 1 ? times[2] - times[1] : 1.0e-6
            dt = min(dt_requested / 10, 1.0e-6)  # Use 10x finer or 1μs, whichever is smaller
        end

        # Define the coupled ODE system with time-varying L₂(t) and M(t)
        function coupled_ode_ML(t, I1, I2)
            M_t = itp_M(t)
            L2_t = itp_L_plasma(t)

            # Calculate time derivatives numerically using small time step
            dt_small = 1.0e-8
            if t + dt_small <= times[end]
                dM_dt = (itp_M(t + dt_small) - M_t) / dt_small
                dL2_dt = (itp_L_plasma(t + dt_small) - L2_t) / dt_small
            else
                dM_dt = (M_t - itp_M(t - dt_small)) / dt_small
                dL2_dt = (L2_t - itp_L_plasma(t - dt_small)) / dt_small
            end

            # System matrix L(t) = [L1 M(t); M(t) L₂(t)]
            # Right-hand side includes both resistance and inductance derivative terms:
            # RHS = [V1 - R1*I1 - dM/dt*I2; V2 - R2*I2 - dM/dt*I1 - dL₂/dt*I2]

            L_matrix = [L1 M_t; M_t L2_t]
            rhs = [
                V1 - R1 * I1 - dM_dt * I2;
                V2 - R2 * I2 - dM_dt * I1 - dL2_dt * I2
            ]

            # Solve L(t) * [dI1/dt; dI2/dt] = rhs
            try
                dI_dt = L_matrix \ rhs
                return dI_dt[1], dI_dt[2]  # dI1/dt, dI2/dt
            catch
                # If matrix is singular, return zero derivatives
                return 0.0, 0.0
            end
        end

        # Initialize solution arrays for output times
        I1_solution = zeros(length(times))
        I2_solution = zeros(length(times))

        # Set initial conditions
        I1_solution[1] = I1_0
        I2_solution[1] = I2_0

        # Current state for integration
        t_current = times[1]
        I1_current = I1_0
        I2_current = I2_0

        # Track which output time we're targeting next
        target_index = 2

        # Integrate with fine time steps until we reach the end
        while t_current < times[end] && target_index <= length(times)
            t_target = times[target_index]

            # Integrate until we reach or exceed the target time
            while t_current < t_target && t_current < times[end]
                # Adjust step size if we would overshoot the target
                dt_step = min(dt, t_target - t_current)

                # 4th-order Runge-Kutta step
                k1_1, k1_2 = coupled_ode_ML(t_current, I1_current, I2_current)
                k2_1, k2_2 = coupled_ode_ML(t_current + dt_step / 2, I1_current + dt_step * k1_1 / 2, I2_current + dt_step * k1_2 / 2)
                k3_1, k3_2 = coupled_ode_ML(t_current + dt_step / 2, I1_current + dt_step * k2_1 / 2, I2_current + dt_step * k2_2 / 2)
                k4_1, k4_2 = coupled_ode_ML(t_current + dt_step, I1_current + dt_step * k3_1, I2_current + dt_step * k3_2)

                # Update currents
                I1_current += dt_step / 6 * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
                I2_current += dt_step / 6 * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

                # Update time
                t_current += dt_step
            end

            # Store result at target time (with interpolation if needed)
            if abs(t_current - t_target) < 1.0e-12  # Exact match
                I1_solution[target_index] = I1_current
                I2_solution[target_index] = I2_current
            else
                # Linear interpolation for slight overshoots (should be rare with adaptive stepping)
                I1_solution[target_index] = I1_current
                I2_solution[target_index] = I2_current
            end

            target_index += 1
        end

        return I1_solution, I2_solution
    end

    # Extract equivalent circuit parameters for plasma loop
    function extract_plasma_circuit_parameters(RP::RAPID; verbose::Bool = false)
        # Plasma current and total current
        plasma_nodes = RP.G.inVol2D .> 0.5
        I_plasma = sum(RP.plasma.Jϕ[plasma_nodes]) * RP.G.dR * RP.G.dZ

        # Plasma resistance from collision frequency
        # R = (me * νei) / (e² * ne) integrated over plasma volume
        RAPID2D.@unpack ee, me = RP.config.constants
        # Resistance by Coulomb collision
        ue_sat_by_Eext = @. -ee * RP.fields.Eϕ_ext / (me * RP.plasma.ν_ei_eff)
        ue_sat_by_Eext[.!isfinite.(ue_sat_by_Eext)] .= 0.0  # Avoid NaN

        # Estimate circuit parameters
        in_wall_nids = RP.G.nodes.in_wall_nids
        LV_plasma = mean(RP.fields.LV_ext[in_wall_nids])
        R_plasma = LV_plasma ./ sum(-ee * RP.plasma.ne[in_wall_nids] .* ue_sat_by_Eext[in_wall_nids] * RP.G.dR * RP.G.dZ)

        # Plasma geometry for inductance estimate
        major_R = 1.5  # m
        minor_r = 0.3  # m
        YY = 0.0  # initially surface current

        μ0 = 4π * 1.0e-7  # H/m
        L_plasma = μ0 * major_R * (log(8 * major_R / minor_r) - 2 + 0.25 * YY)

        if verbose
            println("  Plasma circuit parameters:")
            println("    Self-inductance: $(L_plasma * 1.0e6) μH")
            println("    Resistance: $(R_plasma * 1.0e3) mΩ")
            println("    L/R time: $(L_plasma / R_plasma * 1.0e6) μs")
            println("    Current: $(I_plasma) A")
        end

        return L_plasma, R_plasma, LV_plasma
    end

    # Analyze plasma-coil coupling results and compare with the analytical solutions.
    # The "✓ PASS" / "✗ FAIL" lines printed under `verbose` are cosmetic; the real
    # assertions and thresholds live in the @testitem body.
    function analyze_coupling_results(
            RP::RAPID; verbose::Bool = false, visualize::Bool = false,
            outdir::AbstractString = mktempdir(; cleanup = false)
        )
        if verbose
            println("\n" * "="^70)
            println("PLASMA-COIL COUPLING ANALYSIS")
            println("="^70)
        end

        # Extract time series data
        times = RP.diagnostics.snaps0D.time_s
        I_plasma = RP.diagnostics.snaps0D.I_tor  # Plasma current

        # Extract coil current (assume single coil)
        I_coil = [s.coils_I[1] for s in RP.diagnostics.snaps0D]

        # Get circuit parameters
        coil = RP.coil_system.coils[1]
        L_coil = coil.self_inductance
        R_coil = coil.resistance
        V_coil = coil.voltage_ext

        # Extract plasma parameters (time-averaged for this simple analysis)
        L_plasma, R_plasma, LV_plasma = extract_plasma_circuit_parameters(RP; verbose)

        # Estimate mutual inductance (geometric calculation)
        snaps2D_time_s = RP.diagnostics.snaps2D.time_s
        snaps2D_time_s = range(snaps2D_time_s[1], stop = snaps2D_time_s[end], length = length(snaps2D_time_s))
        M_values = [(2π * RP.coil_system.Green_grid2coils * s.Jϕ[:] / sum(s.Jϕ))[1] for s in RP.diagnostics.snaps2D]
        M_values[1] = M_values[2] # Avoid zero M

        itp_M = cubic_interp(snaps2D_time_s, M_values)

        mean_M = mean(M_values)

        if verbose
            println("\nCircuit Parameters:")
            println("  Coil: L=$(L_coil * 1.0e6)μH, R=$(R_coil * 1.0e3)mΩ, V=$(V_coil)V")
            println("  Plasma: L=$(L_plasma * 1.0e6)μH, R=$(R_plasma * 1.0e3)mΩ")
            println("  Mutual inductance (est.): $(mean_M * 1.0e6)μH")
            println("  Coupling coefficient: $(mean_M / sqrt(L_coil * L_plasma))")
        end

        # Calculate analytical solution
        I_coil_analytical, I_plasma_analytical, λ1, λ2, I_steady =
            calculate_analytical_solution(
            times, L_coil, L_plasma, mean_M,
            R_coil, R_plasma, V_coil, LV_plasma
        )

        # Calculate analytical solution with both time-varying M(t) and L₂(t).
        L_plasma_values = RP.diagnostics.snaps0D.self_inductance_plasma
        L_plasma_values[1] = L_plasma_values[2] # Avoid zero at t=0
        itp_L_plasma = linear_interp(times, L_plasma_values)

        I_coil_analytical_ML, I_plasma_analytical_ML =
            calculate_analytical_solution_time_varying_M_and_L(
            times, L_coil, itp_L_plasma, itp_M,
            R_coil, R_plasma, V_coil, LV_plasma
        )

        if verbose
            println("\nAnalytical Solution (constant M):")
            println("  Eigenvalues: λ₁=$(λ1) s⁻¹, λ₂=$(λ2) s⁻¹")
            println("  Time constants: τ₁=$(-1 / λ1 * 1.0e3)ms, τ₂=$(-1 / λ2 * 1.0e3)ms")
            println("  Steady-state currents: I_coil=$(I_steady[1])A, I_plasma=$(I_steady[2])A")

            println("\nAnalytical Solution (time-varying M & L₂):")
            println("  L₂(t) range: $(minimum(L_plasma_values) * 1.0e6) to $(maximum(L_plasma_values) * 1.0e6) μH")
            println("  L₂ increase: $(((maximum(L_plasma_values) - minimum(L_plasma_values)) / minimum(L_plasma_values) * 100))%")
            println("  Final I_coil (time-varying M&L): $(I_coil_analytical_ML[end]) A")
            println("  Final I_plasma (time-varying M&L): $(I_plasma_analytical_ML[end]) A")
        end

        # Calculate errors (excluding first index to avoid division by zero/Inf issues)
        # Use indices 2:end to avoid t=0 analytical solution issues
        n_start = max(2, length(times) ÷ 4)  # Start from 2nd index or 1/4 of simulation
        n_final = length(times) ÷ 2  # Use second half for main comparison

        # Extract final portions for error calculation (avoiding first index)
        I_coil_sim_final = I_coil[n_final:end]
        I_plasma_sim_final = I_plasma[n_final:end]
        I_coil_ana_final = I_coil_analytical[n_final:end]
        I_plasma_ana_final = I_plasma_analytical[n_final:end]

        # Time-varying M&L analytical solution comparison
        I_coil_ana_ML_final = I_coil_analytical_ML[n_final:end]
        I_plasma_ana_ML_final = I_plasma_analytical_ML[n_final:end]

        # Error metrics with robust calculation (avoid Inf/NaN)
        function calculate_robust_error(sim, ana)
            # Filter out cases where analytical solution is very small (< 1e-10)
            valid_indices = abs.(ana) .> 1.0e-10
            if sum(valid_indices) == 0
                return 0.0  # If no valid points, return zero error
            end
            sim_valid = sim[valid_indices]
            ana_valid = ana[valid_indices]
            return mean(abs.(sim_valid - ana_valid) ./ abs.(ana_valid))
        end

        # Error metrics for constant M solution
        coil_error = calculate_robust_error(I_coil_sim_final, I_coil_ana_final)
        plasma_error = calculate_robust_error(I_plasma_sim_final, I_plasma_ana_final)

        # Error metrics for time-varying M&L solution
        coil_error_ML = calculate_robust_error(I_coil_sim_final, I_coil_ana_ML_final)
        plasma_error_ML = calculate_robust_error(I_plasma_sim_final, I_plasma_ana_ML_final)

        if verbose
            println("\nAccuracy Assessment:")
            println("  === vs Constant M Analytical ===")
            println(@sprintf("  Coil current mean error: %.1f%%", 100 * coil_error))
            println(@sprintf("  Plasma current mean error: %.1f%%", 100 * plasma_error))

            println("  === vs Time-varying M & L₂ Analytical ===")
            println(@sprintf("  Coil current mean error: %.1f%%", 100 * coil_error_ML))
            println(@sprintf("  Plasma current mean error: %.1f%%", 100 * plasma_error_ML))

            if coil_error < 0.1 && plasma_error < 0.2  # Looser tolerance for plasma
                println("  ✓ PASS: Good agreement with constant M analytical solution")
            else
                println("  ✗ FAIL: Poor agreement with constant M analytical solution")
            end

            if coil_error_ML < 0.15 && plasma_error_ML < 0.25  # Slightly looser tolerance for more complex case
                println("  ✓ PASS: Good agreement with time-varying M & L₂ analytical solution")
            else
                println("  ✗ FAIL: Poor agreement with time-varying M & L₂ analytical solution")
            end
        end

        # Create visualization
        if visualize
            create_coupling_plots_combined(
                RP, times, I_coil, I_plasma,
                I_coil_analytical, I_plasma_analytical,
                I_coil_analytical_ML, I_plasma_analytical_ML; outdir
            )
        end

        return (
            coil_error = coil_error,
            plasma_error = plasma_error,
            coil_error_ML = coil_error_ML,
            plasma_error_ML = plasma_error_ML,
            times = times,
            I_coil_sim = I_coil,
            I_plasma_sim = I_plasma,
            I_coil_analytical = I_coil_analytical,
            I_plasma_analytical = I_plasma_analytical,
            I_coil_analytical_ML = I_coil_analytical_ML,
            I_plasma_analytical_ML = I_plasma_analytical_ML,
            M_values = M_values,
            L_plasma_values = L_plasma_values,
            L_coil = L_coil,
            L_plasma = L_plasma,
            mean_M = mean_M,
        )
    end

    # Plot the simulation against the constant-M and time-varying-M&L analytical
    # solutions. `outdir` MUST be unique per run: the filename timestamp is only
    # second-resolution, so two runs sharing a directory can overwrite each other.
    function create_coupling_plots_combined(
            RP, times, I_coil_sim, I_plasma_sim,
            I_coil_ana, I_plasma_ana,
            I_coil_ana_ML, I_plasma_ana_ML;
            outdir::AbstractString = mktempdir(; cleanup = false)
        )
        # Convert time to milliseconds for plotting
        times_ms = times * 1.0e3

        # Current evolution plots - Coil
        p1 = plot(
            times_ms, I_coil_sim,
            label = "Simulation",
            linewidth = 3,
            color = :blue,
            xlabel = "Time (ms)",
            ylabel = "Current (A)",
            title = "Coil Current Evolution"
        )
        plot!(
            p1, times_ms, I_coil_ana,
            label = "Constant M&L",
            linestyle = :dash,
            linewidth = 2,
            color = :red
        )
        plot!(
            p1, times_ms, I_coil_ana_ML,
            label = "Time-varying M&L",
            linestyle = :dashdot,
            linewidth = 2,
            color = :orange
        )

        # Current evolution plots - Plasma
        p2 = plot(
            times_ms, I_plasma_sim,
            label = "Simulation",
            linewidth = 3,
            color = :blue,
            xlabel = "Time (ms)",
            ylabel = "Current (A)",
            title = "Plasma Current Evolution"
        )
        plot!(
            p2, times_ms, I_plasma_ana,
            label = "Constant M&L",
            linestyle = :dash,
            linewidth = 2,
            color = :red
        )
        plot!(
            p2, times_ms, I_plasma_ana_ML,
            label = "Time-varying M&L",
            linestyle = :dashdot,
            linewidth = 2,
            color = :orange
        )

        # Error plots with robust calculation (avoiding first index issues)
        function calculate_error_robust(sim, ana)
            # Use indices 2:end to avoid t=0 issues where ana might be zero
            valid_indices = 2:length(ana)
            ana_valid = ana[valid_indices]
            sim_valid = sim[valid_indices]

            # Filter out very small analytical values to avoid Inf
            large_enough = abs.(ana_valid) .> 1.0e-10
            if sum(large_enough) == 0
                return zeros(length(valid_indices))
            end

            error_values = zeros(length(valid_indices))
            error_values[large_enough] = abs.(sim_valid[large_enough] - ana_valid[large_enough]) ./ abs.(ana_valid[large_enough]) * 100
            return error_values
        end

        coil_error_const = calculate_error_robust(I_coil_sim, I_coil_ana)
        coil_error_ML = calculate_error_robust(I_coil_sim, I_coil_ana_ML)

        plasma_error_const = calculate_error_robust(I_plasma_sim, I_plasma_ana)
        plasma_error_ML = calculate_error_robust(I_plasma_sim, I_plasma_ana_ML)

        # Error plot - Coil
        p3 = plot(
            times_ms[2:end], coil_error_const,
            label = "vs Constant M&L",
            linewidth = 2,
            color = :red,
            xlabel = "Time (ms)",
            ylabel = "Error (%)",
            title = "Coil Current Errors"
        )
        plot!(
            p3, times_ms[2:end], coil_error_ML,
            label = "vs Time-varying M&L",
            linewidth = 2,
            color = :orange
        )

        # Error plot - Plasma
        p4 = plot(
            times_ms[2:end], plasma_error_const,
            label = "vs Constant M&L",
            linewidth = 2,
            color = :red,
            xlabel = "Time (ms)",
            ylabel = "Error (%)",
            title = "Plasma Current Errors"
        )
        plot!(
            p4, times_ms[2:end], plasma_error_ML,
            label = "vs Time-varying M&L",
            linewidth = 2,
            color = :orange
        )

        # Combined plot
        plot_combined = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 800))

        # Save plot into the caller-provided unique directory
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
        full_path = joinpath(outdir, "plasma_coil_coupling_comparison_$(timestamp).png")
        savefig(plot_combined, full_path)
        @info "Plasma-coil coupling plot saved" path = full_path

        # Create animation if possible
        anim_path = joinpath(outdir, "plasma_coil_coupling_$(timestamp).mp4")
        try
            animate_snaps2D(
                RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                [:ne, :ue_para, :Jϕ, :E_para_tot];
                wall = RP.fitted_wall,
                filename = anim_path
            )
            @info "Plasma-coil coupling animation saved" path = anim_path
        catch e
            @warn "Plasma-coil coupling animation creation failed" exception = e
        end

        return plot_combined
    end
end

@testitem "Plasma-Coil Inductive Coupling" tags = [:regression] setup = [RegressionCommon, CouplingAnalysis] begin
    # Declared here as well as in the snippets: a snippet becomes a module under the
    # ReTestItems path, and `using` only re-exports names a module OWNS, so imports made
    # inside a snippet are invisible to this body.
    using RAPID2D.Statistics
    using Printf

    verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"
    visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

    # Artifacts go to a fresh per-run directory; cleanup=false so they outlive the
    # process and stay inspectable. The path is reported by @info when plots are saved.
    outdir = visualize ? mktempdir(; cleanup = false) : tempname()

    # ── Discretization ──────────────────────────────────────────────────────────
    # The snapshot intervals fix how many points the comparison spans: 101 × 0D samples
    # feed the current error metrics, 21 × 2D samples feed the mutual inductance M(t).
    config = regression_config(;
        NR = 40,
        NZ = 60,
        dt = 5.0e-6,                     # 5 μs
        t_end_s = 2.0e-3,                # 2 ms
        R0B0 = 2.5,                    # T⋅m
        prefilled_gas_pressure = 0.0,  # no neutral gas ⇒ no ionization source
        snap0D_Δt_s = 20.0e-6,
        snap2D_Δt_s = 100.0e-6,
    )

    RP = RAPID{Float64}(config)

    RP.flags = SimulationFlags{Float64}(
        # ── THE PHYSICS UNDER TEST: inductive plasma ↔ coil coupling ────────────
        # Ampère solves for the field of the plasma current; E_para_self_EM feeds the
        # resulting -∂A/∂t back onto the electrons. Together they ARE the inductive
        # link between the plasma loop and the coil circuit — this test measures it.
        Ampere = true,
        E_para_self_EM = true,
        Ampere_Itor_threshold = 0.0,  # no dead band: couple from the first amp onwards

        # The plasma responds through parallel momentum only, and its loop resistance
        # comes from the collision frequencies, so I_plasma(t) is a pure LR response.
        ud_evolve = true,             # the ONLY evolved plasma quantity
        Atomic_Collision = true,      # electron-neutral drag
        Coulomb_Collision = true,     # ν_ei ⇒ the plasma-loop resistance

        # ── DELIBERATELY DISABLED: a frozen, stationary plasma column ───────────
        # Every one of these would move, reshape or refuel the current channel, and
        # the two-coupled-LR-circuits analytical solution would no longer apply.
        Te_evolve = false,            # fixed temperature ⇒ fixed resistivity
        Ti_evolve = false,
        src = false,
        convec = false,               # no convection (stationary plasma)
        diffu = false,                # no diffusion
        Gas_evolve = false,
        update_ni_independently = false,
        Include_ud_convec_term = false,
        Include_ud_diffu_term = false,
        Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        E_para_self_ES = false,       # electrostatic self-field is not what is tested
        Global_JxB_Force = false,     # no bulk displacement of the column

        # ── Numerics ───────────────────────────────────────────────────────────
        Implicit = true,
        FLF_nstep = 50,
        Damp_Transp_outWall = true,
        negative_n_correction = true,
    )

    initialize!(RP)

    # Pure toroidal field with E0 = 0.0 — no external Eϕ imposed by this scenario.
    # NOTE: E0 = 0.0 does NOT mean undriven. `initialize!` above already installed the
    # manual-device default loop voltage (src/initialization.jl, set_RZ_B_E_manually!:
    # 0.3 V/m); this helper overwrites Eϕ / Eϕ_ext / E_para_ext but NOT fields.LV_ext,
    # and that surviving LV_ext is what drives the plasma loop. It is read back as V₂
    # (and, via the saturated drift, as R_plasma) in extract_plasma_circuit_parameters.
    setup_toroidal_field!(RP; E0 = 0.0, verbose)

    # ── Initial plasma: a stationary top-hat current channel ────────────────────
    # The background floor outside the core is cold and rarefied enough to carry no
    # appreciable current, but nonzero so that ν_ei and the transport coefficients stay
    # finite across the whole grid.
    cenR, radius = 1.5, 0.3
    RP.plasma.ne .= tophat_blob(RP.G; cenR, radius, n0 = 1.0e16, background = 1.0e6)
    RP.plasma.ni .= tophat_blob(RP.G; cenR, radius, n0 = 1.0e16, background = 1.0e6)
    RP.plasma.Te_eV .= tophat_blob(RP.G; cenR, radius, n0 = 10.0, background = 1.0)
    RP.plasma.Ti_eV .= tophat_blob(RP.G; cenR, radius, n0 = 0.026, background = 0.1)

    # Start from rest: no parallel flow and no toroidal current, so the entire current
    # trace that is compared against the analytical solution is generated by the run.
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)
    fill!(RP.plasma.Jϕ, 0.0)

    # ── External coil: circuit 1 of the coupled pair ────────────────────────────
    # A single filament inboard of the column (R = 1.5 m) and well off the midplane, so
    # it links part of the column's flux without sitting inside it.
    coil_r = 1.2                 # m, coil major radius
    coil_z = 0.8                 # m, above the midplane
    coil_area = π * (0.05)^2     # m², conductor cross-section (5 cm radius)

    # Copper resistance around the loop, and the standard single-turn circular-loop
    # self-inductance with internal-inductance factor YY = 1.
    resistivity = 1.68e-8        # Ω⋅m (copper)
    coil_resistance = resistivity * 2π * coil_r / coil_area
    μ0 = 4π * 1.0e-7
    YY = 1.0
    coil_self_L = μ0 * coil_r * (log(8 * coil_r / sqrt(coil_area / π)) - 2 + 0.25 * YY)

    # NOTE the trailing 0.0 applied voltage: the coil is NOT driven. Its current is
    # induced entirely by the plasma loop, which is exactly the quantity under test.
    add_coil!(
        RP.coil_system, Coil(
            (r = coil_r, z = coil_z),
            coil_area,               # effective area
            coil_resistance,
            coil_self_L,
            true,                    # is_powered
            true,                    # is_controllable
            "coupling_coil",
            50.0,                    # max_voltage [V]
            10.0e3,                    # max_current [A]
            0.0,                     # initial current [A]
            0.0                      # applied voltage [V] — no external drive
        )
    )
    initialize_coil_system!(RP)   # rebuild the coil-system matrices with the new coil

    run_simulation!(RP)

    results = analyze_coupling_results(RP; verbose, visualize, outdir)

    # Test assertions
    @test results !== nothing  # Should return valid results

    if results !== nothing
        # TODO: need to reduce error further
        @test results.coil_error_ML < 0.1      # Coil error should be reasonable
        @test results.plasma_error_ML < 0.1    # Plasma error (looser tolerance)
        @test results.L_coil > 0              # Positive inductance
        @test results.L_plasma > 0            # Positive inductance
        @test abs(results.mean_M) > 0     # Non-zero mutual inductance
        @test length(results.times) > 1       # Multiple time points

        # Check coupling coefficient is physical
        k = results.mean_M / sqrt(results.L_coil * results.L_plasma)
        @test abs(k) < 1.0  # Coupling coefficient must be less than 1
    end
end
