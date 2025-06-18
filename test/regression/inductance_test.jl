#!/usr/bin/env julia
"""
Basic Inductance Test for RAPID2D.jl

This script tests the basic inductance calculations and validates them against
analytical solutions for a simple single-filament circuit model.

"""

using RAPID2D
using RAPID2D.Statistics
using RAPID2D.Interpolations
using Test
using Printf

# Environment variable controls
verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"
visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

# Only load Plots if visualization is requested
if visualize
    using Plots
    using Dates
end


function run_inductance_test(; verbose=false, visualize=false)
    if verbose
        println("=" ^ 60)
        println("RAPID2D.jl Basic Inductance Test")
        println("=" ^ 60)
    end

    # Create configuration
    config = create_basic_config()

    # Initialize RAPID simulation
    if verbose
        println("Initializing RAPID simulation...")
    end
    RP = RAPID{Float64}(config)

    # Set up flags
    setup_flags!(RP)

    # Initialize the simulation
    initialize!(RP)

    # Set up magnetic field and plasma after initialization
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)

    # Set time parameters
    RP.t_end_s = 20e-3

    # Run simulation
    if verbose
        println("Running simulation...")
    end
    run_simulation!(RP)

    # Return RP for testing
    return RP
end

function create_basic_config()
    """Create basic configuration for inductance testing"""

    config = SimulationConfig{Float64}()

    # Grid parameters
    config.NR = 30
    config.NZ = 50

    # Physical parameters
    config.prefilled_gas_pressure = 0.0  # Pa
    config.R0B0 = 3.0  # Tesla⋅meter

    config.dt = 25e-6
    config.snap0D_Δt_s = 50e-6
    config.snap2D_Δt_s = 500e-6

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_flags!(RP::RAPID)
    """Set up simulation flags for inductance testing"""

    # Basic flags for inductance test
    RP.flags = SimulationFlags{Float64}(
        Atomic_Collision = false,
        Te_evolve = false,
        Ti_evolve = false,
        src = false,
        convec = false,
        diffu = false,
        ud_evolve = true,
        Include_ud_convec_term = false,
        Include_ud_diffu_term = false,
        Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        update_ni_independently = false,
        Gas_evolve = false,
        Ampere = true,
        E_para_self_ES = false,
        negative_n_correction = true
    )

    # Additional flags for inductance testing
    RP.flags.Coulomb_Collision = true
    RP.flags.E_para_self_EM = true
    RP.flags.Ampere_Itor_threshold = 0.0
    RP.flags.FLF_nstep = 100
    RP.flags.Implicit = true
    RP.flags.Damp_Transp_outWall = true
    RP.flags.Global_JxB_Force = false  # Not needed for basic inductance test

end

function setup_magnetic_field!(RP::RAPID; verbose::Bool=false)
    """Set up pure toroidal magnetic field configuration"""

    # Zero poloidal field components (pure toroidal field)
    fill!(RP.fields.BR, 0.0)
    fill!(RP.fields.BZ, 0.0)
    fill!(RP.fields.BR_ext, 0.0)
    fill!(RP.fields.BZ_ext, 0.0)

    # Set Jacobian for toroidal geometry
    @. RP.G.Jacob = RP.G.R2D

    # Set toroidal field: Bφ = R₀B₀/R
    @. RP.fields.Bϕ = RP.config.R0B0 / RP.G.Jacob

    # Update total field
    @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
    @. RP.fields.Btot = abs(RP.fields.Bϕ)

    # Update unit vectors
    @. RP.fields.bR = RP.fields.BR / RP.fields.Btot
    @. RP.fields.bZ = RP.fields.BZ / RP.fields.Btot
    @. RP.fields.bϕ = RP.fields.Bϕ / RP.fields.Btot

    # Set up electric field
    E0 = 0.3  # V/m
    @. RP.fields.Eϕ = E0 * mean(RP.G.R1D) / RP.G.Jacob
    @. RP.fields.Eϕ_ext = RP.fields.Eϕ
    @. RP.fields.E_para_ext = RP.fields.Eϕ * RP.fields.bϕ

    if verbose
        println("  ✓ Magnetic field configuration set")
        println("    Pure toroidal field with R₀B₀ = $(RP.config.R0B0) T⋅m")
    end
end

function setup_plasma!(RP::RAPID; verbose::Bool=false)
    """Set up initial plasma density distribution"""

    # Plasma parameters (similar to MATLAB test)
    cenR = mean(RP.G.R1D)  # Center R
    cenZ = mean(RP.G.Z1D)  # Center Z
    cenR = 1.5  # m
    cenZ = 0.0  # m
    radius = 0.3  # m
    n0 = 1e16  # m⁻³

    # Create Gaussian plasma profile
    # ini_n = @. n0 * exp(-(((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) / (2 * radius^2)))
    ini_n = n0 * ones(size(RP.G.R2D))  # Initialize with zeros

    # Apply spatial mask (only inside minor radius)
    mask = @. sqrt((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) < radius
    ini_n .*= mask

    # Set plasma densities
    @. RP.plasma.ne = ini_n
    @. RP.plasma.ni = ini_n

    # Set temperatures
    fill!(RP.plasma.Te_eV, 10.0)  # eV
    fill!(RP.plasma.Ti_eV, 0.03)  # eV (room temperature)

    # Zero initial velocities
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)

    if verbose
        println("  ✓ Initial plasma configuration set")
        println("    Center: R=$(cenR)m, Z=$(cenZ)m")
        println("    Radius: $(radius)m")
        println("    Peak density: $(n0/1e16) × 10¹⁶ m⁻³")
    end
end

function analyze_results(RP::RAPID; verbose::Bool=false, visualize::Bool=false)
    """Analyze simulation results and compare with analytical solution"""

    if verbose
        println("\n" * "=" ^ 60)
        println("INDUCTANCE ANALYSIS")
        println("=" ^ 60)
    end

    # Extract time series data
    times = RP.diagnostics.snaps0D.time_s
    I_tor = RP.diagnostics.snaps0D.I_tor

    RAPID2D.@unpack ee, me = RP.config.constants
    # Resistance by Coulomb collision
    ue_sat_by_Evac = @. -ee*RP.fields.Eϕ_ext/(me * RP.plasma.ν_ei_eff);
    ue_sat_by_Evac[.!isfinite.(ue_sat_by_Evac)] .= 0.0  # Avoid NaN

    # Estimate circuit parameters
    in_wall_nids = RP.G.nodes.in_wall_nids
    LV_estimate = mean(RP.fields.LV_ext[in_wall_nids])
    R_estimate = LV_estimate./sum(-ee*RP.plasma.ne[in_wall_nids].*ue_sat_by_Evac[in_wall_nids]*RP.G.dR*RP.G.dZ);

    # Plasma geometry for inductance estimate
    major_R = 1.5  # m
    minor_r = 0.3  # m
    Y = 1.0  # Internal inductance factor

    μ0 = 4π * 1e-7  # H/m
    L_estimate = μ0 * major_R * (log(8 * major_R / minor_r) - 2 + 0.25 * Y)

    # Estimate mutual inductance (geometric calculation)
	snap0D_time_s = RP.diagnostics.snaps0D.time_s
	snap0D_time_s = range(snap0D_time_s[1], stop=snap0D_time_s[end], length=length(snap0D_time_s))
    L_values = RP.diagnostics.snaps0D.L_self_plasma
    itp_L_self_plasma = cubic_spline_interpolation(snap0D_time_s, L_values)

    # L/R time constant
    tau_LR = L_estimate / R_estimate

    if verbose
        println("Circuit Parameter Estimates:")
        println(@sprintf("  Loop voltage: %.3f V", LV_estimate))
        println(@sprintf("  Final current: %.3f A", I_tor[end]))
        println(@sprintf("  Resistance: %.6f Ω", R_estimate))
        println(@sprintf("  Inductance: %.6f μH", L_estimate * 1e6))
        println(@sprintf("  L/R time: %.1f μs", tau_LR * 1e6))
    end

    # Analytical solution (constant L approximation)
    I_sat_analytical = LV_estimate / R_estimate
    I_analytical = @. I_sat_analytical * (1 - exp(-times / tau_LR))

    # More accurate analytical solution with time-varying inductance
    I_analytical_timevar = calculate_time_varying_inductance_solution(times, LV_estimate, R_estimate, itp_L_self_plasma)

    # Create plots
    if visualize
        create_inductance_plots(RP, times, I_tor, I_analytical, I_analytical_timevar)
    end

    # Calculate error metrics and return for testing
    if length(I_tor) == length(I_analytical) && length(I_tor) == length(I_analytical_timevar)
        # Error vs constant L analytical solution
        relative_error = abs.(I_tor - I_analytical) ./ (abs.(I_analytical) )
        relative_error[.!isfinite.(relative_error)] .= 0.0  # Avoid NaN & Inf
        mean_error = mean(relative_error)
        max_error = maximum(relative_error)

        # Error vs time-varying L analytical solution
        relative_error_timevar = abs.(I_tor - I_analytical_timevar) ./ (abs.(I_analytical_timevar) )
        relative_error_timevar[.!isfinite.(relative_error_timevar)] .= 0.0  # Avoid NaN & Inf
        mean_error_timevar = mean(relative_error_timevar)
        max_error_timevar = maximum(relative_error_timevar)

        if verbose
            println("\nAccuracy Assessment:")
            println("  === vs Constant L Analytical ===")
            println(@sprintf("  Mean relative error: %.2f%%", 100*mean_error))
            println(@sprintf("  Max relative error: %.2f%%", 100*max_error))

            println("  === vs Time-varying L Analytical ===")
            println(@sprintf("  Mean relative error: %.2f%%", 100*mean_error_timevar))
            println(@sprintf("  Max relative error: %.2f%%", 100*max_error_timevar))

            if mean_error < 0.03
                println("  ✓ PASS: Good agreement with constant L analytical solution")
            else
                println("  ✗ FAIL: Poor agreement with constant L analytical solution")
            end

            if mean_error_timevar < 0.025  # Slightly tighter tolerance for more accurate model
                println("  ✓ PASS: Good agreement with time-varying L analytical solution")
            else
                println("  ✗ FAIL: Poor agreement with time-varying L analytical solution")
            end
        end

        return (mean_error=mean_error, max_error=max_error,
                mean_error_timevar=mean_error_timevar, max_error_timevar=max_error_timevar,
                times=times, I_tor=I_tor, I_analytical=I_analytical, I_analytical_timevar=I_analytical_timevar)
    else
        return nothing
    end
end

function create_inductance_plots(RP, times, I_sim, I_analytical, I_analytical_timevar)
    """Create visualization plots for inductance test results"""

    # Current evolution plot
    p1 = plot(times * 1e3, I_sim,
              label="Simulation",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Toroidal Current (A)",
              title="L/R Circuit Response")

    plot!(p1, times * 1e3, I_analytical,
          label="Analytical (const L)",
          linestyle=:dash,
          linewidth=2)

    plot!(p1, times * 1e3, I_analytical_timevar,
          label="Analytical (time-var L)",
          linestyle=:dot,
          linewidth=2,
          color=:green)

    # Error plots
    if length(I_sim) == length(I_analytical) && length(I_sim) == length(I_analytical_timevar)
        error_const_L = abs.(I_sim - I_analytical) ./ (abs.(I_analytical)) * 100
        error_timevar_L = abs.(I_sim - I_analytical_timevar) ./ (abs.(I_analytical_timevar)) * 100

        p2 = plot(times * 1e3, error_const_L,
                  label="vs Const L",
                  linewidth=2,
                  xlabel="Time (ms)",
                  ylabel="Error (%)",
                  title="Simulation Errors")

        plot!(p2, times * 1e3, error_timevar_L,
              label="vs Time-var L",
              linestyle=:dash,
              linewidth=2,
              color=:green)
    else
        p2 = plot(title="Error analysis unavailable")
    end

    # Inductance evolution plot
    snap0D_times = RP.diagnostics.snaps0D.time_s
    L_values = RP.diagnostics.snaps0D.L_self_plasma

    p3 = plot(snap0D_times * 1e3, L_values * 1e6,
              label="L_self_plasma(t)",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Self-Inductance (μH)",
              title="Time-Varying Plasma Inductance")

    # Combined plot
    plot_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 900))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = "inductance_test_$(timestamp).png"
    full_path = joinpath(pwd(), filename)
    savefig(plot_combined, full_path)

    println("\nVisualization:")
    println("  ✓ Plot saved as: $(full_path)")

    # Create animation if possible
    try
        animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                    [:ne,:ue_para,:Jϕ,:E_para_tot];
                    wall=RP.fitted_wall,
                    filename="inductance_test_snaps2D_$(timestamp).mp4")
        println("  ✓ Animation saved as: inductance_test_snaps2D_$(timestamp).mp4")
    catch e
        println("  ⚠ Animation creation failed: $(e)")
    end

    return plot_combined
end

function calculate_time_varying_inductance_solution(times, V, R, itp_L; I0=0.0)
    """
    Calculate analytical solution for RL circuit with time-varying inductance L(t)

    Solves: L(t) * dI/dt + dL/dt * I + R * I = V

    Using numerical integration with 4th-order Runge-Kutta method
    """

    dt = length(times) > 1 ? times[2] - times[1] : 1e-6
    I_solution = zeros(length(times))
    I_solution[1] = I0

    for i in 2:length(times)
        t_prev = times[i-1]
        t_curr = times[i]
        I_prev = I_solution[i-1]

        # Define the ODE: dI/dt = (V - R*I - dL/dt*I) / L(t)
        function ode_func(t, I)
            L_t = itp_L(t)

            # Calculate dL/dt numerically
            dt_small = 1e-8
            if t + dt_small <= times[end]
                dL_dt = (itp_L(t + dt_small) - L_t) / dt_small
            else
                dL_dt = (L_t - itp_L(t - dt_small)) / dt_small
            end

            # dI/dt = (V - R*I - dL/dt*I) / L(t)
            if L_t > 0
                return (V - R*I - dL_dt*I) / L_t
            else
                return 0.0
            end
        end

        # 4th-order Runge-Kutta integration
        k1 = ode_func(t_prev, I_prev)
        k2 = ode_func(t_prev + dt/2, I_prev + dt*k1/2)
        k3 = ode_func(t_prev + dt/2, I_prev + dt*k2/2)
        k4 = ode_func(t_curr, I_prev + dt*k3)

        I_solution[i] = I_prev + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    end

    return I_solution
end

@testset "RAPID2D.jl Basic Inductance Test" begin
    # Run the inductance test
    RP = run_inductance_test(verbose=verbose, visualize=visualize)

    # Analyze results and extract test metrics
    results = analyze_results(RP; verbose, visualize)

    # Test assertions
    @test results !== nothing  # Should return valid results

    if results !== nothing
        @test results.mean_error < 0.03  # Mean error should be less than 3%
        @test results.max_error < 0.05   # Max error should be less than 5%
        @test results.I_tor[end] > 0     # Final current should be positive
        @test length(results.times) > 1  # Should have multiple time points
    end
end



# Main execution (when run directly as a script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl Basic Inductance Test...")

    try
        RP = run_inductance_test(verbose=true, visualize=true)
        results = analyze_results(RP; verbose=true, visualize=true)

        if results !== nothing && results.mean_error < 0.03
            println("\n" * "=" ^ 60)
            println("✓ INDUCTANCE TEST COMPLETED SUCCESSFULLY")
            println("=" ^ 60)
        else
            println("\n" * "=" ^ 60)
            println("✗ INDUCTANCE TEST FAILED - Poor accuracy")
            println("=" ^ 60)
        end
    catch e
        println("\n" * "=" ^ 60)
        println("✗ INDUCTANCE TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 60)
        rethrow(e)
    end
end
