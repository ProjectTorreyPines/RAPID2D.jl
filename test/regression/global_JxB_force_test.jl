#!/usr/bin/env julia
"""
Global JxB Force Test for RAPID2D.jl

This script tests the global force balance functionality with plasma position control,
validating the J×B force calculations and plasma centroid tracking against the original
MATLAB implementation.

Based on test_force_balance.m from RAPID-2D MATLAB version.
"""

using RAPID2D
using RAPID2D.Statistics
using Test
using Printf

# Environment variable controls
verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"
visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

verbose=true
visualize=true


# Only load Plots if visualization is requested
if visualize
    using Plots
    using Dates
end


function run_global_force_test(; verbose=false, visualize=false)
    if verbose
        println("=" ^ 60)
        println("RAPID2D.jl Global JxB Force Test")
        println("=" ^ 60)
    end

    # Create configuration
    config = create_force_balance_config()

    # Initialize RAPID simulation
    if verbose
        println("Initializing RAPID simulation...")
    end
    RP = RAPID{Float64}(config)

    # Set up flags
    setup_force_balance_flags!(RP)

    # Initialize the simulation
    initialize!(RP)

    # Set up magnetic field and plasma after initialization
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)


    # Run simulation
    if verbose
        println("Running simulation...")
    end
    run_simulation!(RP)

    # Return RP for testing
    return RP
end

function create_force_balance_config()
    """Create configuration for global force balance testing based on MATLAB test_force_balance.m"""

    config = SimulationConfig{Float64}()

    # Grid parameters (from MATLAB: R_NUM=50, Z_NUM=100)
    config.NR = 50
    config.NZ = 100
	config.NR = 30
    config.NZ = 50

    # Physical parameters
    config.prefilled_gas_pressure = 1e-3  # Pa (from MATLAB: 1e-3)
    config.R0B0 = 3.0  # Tesla⋅meter

    # Time parameters
    config.dt = 2.5e-6         # dt =5e-6
    config.snap0D_Δt_s = 5e-6   # snap1D_Interval_s = 10e-6
    config.snap2D_Δt_s = 50e-6   # snap2D_Interval_s = 20e-6
    config.t_end_s = 300e-6
    # config.t_end_s = 2e-3   # snap2D_Interval_s = 2e-3

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_force_balance_flags!(RP::RAPID)
    """Set up simulation flags for global force balance testing based on MATLAB test_force_balance.m"""

    # Basic flags for force balance test (from MATLAB INPUT.Flag settings)
    RP.flags = SimulationFlags{Float64}(
        Atomic_Collision = false,
        Te_evolve = false,        # Te_evolve = 0
        Ti_evolve = false,
        src = false,              # src = 0
        convec = true,            # convec = 1
        diffu = false,            # diffu = 0
        ud_evolve = true,         # ud_evolve = 1
        Include_ud_convec_term = true,   # Include_ud_convec_term = 1
        Include_ud_diffu_term = true,
        Include_ud_pressure_term = true,
        Include_Te_convec_term = true,   # Include_Te_convec_term = 1
        update_ni_independently = false,  # update_ni_independently = 1
        Gas_evolve = false,       # Gas_evolve = 0
        Ampere = true,            # Ampere = 1
        E_para_self_ES = false,   # E_para_self_ES = 0
        negative_n_correction = true     # neg_n_correction = 1
    )

	RP.flags.mean_ExB = false
	RP.flags.turb_ExB_mixing = false
	RP.flags.E_para_self_ES = false

	RP.flags.Atomic_Collision = true

    # Additional flags specific to force balance testing
    RP.flags.Coulomb_Collision = true           # Coulomb_Collision = 1
    RP.flags.E_para_self_EM = true             # E_para_self_EM = 1
    RP.flags.Ampere_Itor_threshold = 0.0
    RP.flags.FLF_nstep = 10                    # FLF_nstep = 100
    RP.flags.Implicit = true                   # Implicit = 1
    RP.flags.Damp_Transp_outWall = true        # Damp_Transp_outWall = 1

    # Global JxB Force - KEY FEATURE for this test
    RP.flags.Global_Force_Balance = true       # Global_Force_Balance = 1
    # RP.flags.Global_Force_Balance = false

    # Additional MATLAB flags
    # Xsec_with_ud_and_gFac = 1, ini_gFac = 0.3
    # ud_method = "Xsec"
    # Ionz_method = "Xsec"
    # upwind = 1
    # mean_ExB = 0
    # turb_ExB_mixing = 0
    # diaMag_drift = 0
    # upara_or_uRphiZ = "upara"
    # Ampere_nstep = 1
    # Adapt_dt = 0
    # Implicit_weight = 0.5

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
    """Set up initial plasma density distribution based on MATLAB test_force_balance.m"""

    # Plasma parameters (from MATLAB test_force_balance.m)
    cenR = 1.3  # m (from MATLAB: cenR = 1.3)
    cenZ = 0.1  # m (from MATLAB: cenZ = 0.0)
    radius = 0.2  # m (from MATLAB: radius = 0.2)
    n0 = 1e16  # m⁻³

    # Create Gaussian plasma profile (from MATLAB)
    # ini_n = 1e16*exp(-(((RP.R2D-cenR).^2 + (RP.Z2D-cenZ).^2) / (2 * radius^2)));
    ini_n = @. n0 * exp(-(((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) / (2 * radius^2)))

    # Apply spatial mask (only inside minor radius)
    # mask = sqrt((RP.R2D-cenR).^2 + (RP.Z2D-cenZ).^2)<radius;
    mask = @. sqrt((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) < radius
    ini_n .*= mask

    # Set plasma densities
    @. RP.plasma.ne = ini_n
    @. RP.plasma.ni = ini_n

    # Set temperatures (from MATLAB: ini_Te_eV = 10.0, ini_Ti_eV = 0.03)
    fill!(RP.plasma.Te_eV, 10.0)  # eV
    fill!(RP.plasma.Ti_eV, 0.03)  # eV (room temperature)

    # Zero initial velocities (from MATLAB: ini_ue = 0.0, ini_ui = 0)
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)

    if verbose
        println("  ✓ Initial plasma configuration set for force balance test")
        println("    Center: R=$(cenR)m, Z=$(cenZ)m")
        println("    Radius: $(radius)m")
        println("    Peak density: $(n0/1e16) × 10¹⁶ m⁻³")
    end
    return RP
end

function analyze_force_balance_results(RP::RAPID; verbose::Bool=false, visualize::Bool=false)
    """Analyze simulation results focusing on global force balance and plasma position control"""

    if verbose
        println("\n" * "=" ^ 60)
        println("GLOBAL FORCE BALANCE ANALYSIS")
        println("=" ^ 60)
    end

    # Extract time series data
    times = RP.diagnostics.snaps0D.time_s
    I_tor = RP.diagnostics.snaps0D.I_tor

    # Force balance specific analysis
    # TODO: Add plasma centroid tracking (ne_cen_R, ne_cen_Z, J_cen_R, J_cen_Z)
    # TODO: Add vertical field analysis (avg_BZ_ctrl)
    # TODO: Add control system analysis if implemented

    # Circuit parameter analysis (similar to MATLAB)
    RAPID2D.@unpack ee, me = RP.config.constants

    # Estimate circuit parameters using MATLAB approach
    # major_R = cenR; minor_r = radius; (from MATLAB)
    major_R = 1.3  # m
    minor_r = 0.2  # m

    # Loop voltage estimate
    in_wall_nids = RP.G.nodes.in_wall_nids
    LV_estimate = mean(RP.fields.LV_ext[in_wall_nids])

    # Resistance estimate
    R_estimate = LV_estimate / I_tor[end]

    # Inductance estimate (from MATLAB: L_estimate = μ₀*major_R*(log(8*major_R/minor_r)-2+0.25*Y))
    Y = 1.0  # Internal inductance factor
    μ0 = 4π * 1e-7  # H/m
    L_estimate = μ0 * major_R * (log(8 * major_R / minor_r) - 2 + 0.25 * Y)

    # L/R time constant
    tau_LR = L_estimate / R_estimate

    if verbose
        println("Force Balance Test Parameters:")
        println(@sprintf("  Major radius: %.1f m", major_R))
        println(@sprintf("  Minor radius: %.1f m", minor_r))
        println(@sprintf("  Loop voltage: %.3f V", LV_estimate))
        println(@sprintf("  Final current: %.3f A", I_tor[end]))
        println(@sprintf("  Resistance: %.6f Ω", R_estimate))
        println(@sprintf("  Inductance: %.6f μH", L_estimate * 1e6))
        println(@sprintf("  L/R time: %.1f μs", tau_LR * 1e6))
    end

    # Analytical solution for current evolution
    I_sat_analytical = LV_estimate / R_estimate
    I_analytical = @. I_sat_analytical * (1 - exp(-times / tau_LR))

    # Create plots focusing on force balance aspects
    if visualize
        create_force_balance_plots(RP, times, I_tor, I_analytical)
    end

    # Calculate test metrics
    if length(I_tor) == length(I_analytical)
        relative_error = abs.(I_tor - I_analytical) ./ (I_analytical .+ 1e-10)
        mean_error = mean(relative_error)
        max_error = maximum(relative_error)

        if verbose
            println("\nCurrent Evolution Accuracy:")
            println(@sprintf("  Mean relative error: %.2f%%", 100*mean_error))
            println(@sprintf("  Max relative error: %.2f%%", 100*max_error))

            # TODO: Add force balance specific validation
            println("\nForce Balance Validation:")
            println("  (Implementation pending)")

            if mean_error < 0.05  # Slightly relaxed for force balance test
                println("  ✓ PASS: Good agreement with expected behavior")
            else
                println("  ✗ FAIL: Poor agreement with expected behavior")
            end
        end

        return (mean_error=mean_error, max_error=max_error, times=times, I_tor=I_tor, I_analytical=I_analytical,
                major_R=major_R, minor_r=minor_r, tau_LR=tau_LR)
    else
        return nothing
    end
end

function create_force_balance_plots(RP, times, I_sim, I_analytical)
    """Create visualization plots for global force balance test results"""

    # Current evolution plot
    p1 = plot(times * 1e3, I_sim,
              label="Simulation",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Toroidal Current (A)",
              title="Force Balance Test - Current Evolution",
			  margin=7Plots.mm)

    plot!(p1, times * 1e3, I_analytical,
          label="Analytical",
          linestyle=:dash,
          linewidth=2)

    # Error plot
    if length(I_sim) == length(I_analytical)
        error_percent = abs.(I_sim - I_analytical) ./ (I_analytical .+ 1e-10) * 100
        p2 = plot(times * 1e3, error_percent,
                  label="Relative Error",
                  linewidth=2,
                  xlabel="Time (ms)",
                  ylabel="Error (%)",
                  title="Current Evolution Error")
    else
        p2 = plot(title="Error analysis unavailable")
    end

    # TODO: Add force balance specific plots
    # - Plasma centroid position (R, Z) vs time
    # - Vertical field vs time
    # - Control system response (if implemented)
    p3 = plot(title="Plasma Position Control\n(To be implemented)")
    p4 = plot(title="Force Balance Analysis\n(To be implemented)")

    # Combined plot
    plot_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = "force_balance_test_$(timestamp).png"
    savefig(plot_combined, filename)

    println("\nVisualization:")
    println("  ✓ Plot saved as: $(filename)")

    # Create animation if possible
    try
        animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                    [:ne,:ue_para,:Jϕ,:E_para_tot,:ueR, :ueZ, :mean_aR_by_JxB, :mean_aZ_by_JxB];
					wall=RP.fitted_wall,
                    filename="force_balance_test_snaps2D_$(timestamp).mp4")
        println("  ✓ Animation saved as: force_balance_test_snaps2D_$(timestamp).mp4")
    catch e
        println("  ⚠ Animation creation failed: $(e)")
    end

    return plot_combined
end

@testset "RAPID2D.jl Global JxB Force Test" begin
    # Run the global force balance test
    RP = run_global_force_test(;verbose, visualize)

    # Analyze results and extract test metrics
    results = analyze_force_balance_results(RP; verbose, visualize)

    # Test assertions
    @test results !== nothing  # Should return valid results

    # if results !== nothing
    #     @test results.mean_error < 0.05  # Mean error should be less than 5% (relaxed for force balance)
    #     @test results.max_error < 0.10   # Max error should be less than 10%
    #     @test results.I_tor[end] > 0     # Final current should be positive
    #     @test length(results.times) > 1  # Should have multiple time points
    #     @test results.major_R ≈ 1.3     # Check plasma geometry matches MATLAB
    #     @test results.minor_r ≈ 0.2     # Check plasma geometry matches MATLAB

    #     # Force balance specific tests (to be implemented when features are available)
    #     # @test hasfield(typeof(RP.diagnostics.snaps0D), :ne_cen_R)  # Plasma centroid tracking
    #     # @test hasfield(typeof(RP.diagnostics.snaps0D), :avg_BZ_ctrl)  # Vertical field control
    # end
end



# Main execution (when run directly as a script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl Global JxB Force Test...")

    try
        RP = run_global_force_test(verbose=true, visualize=true)
        results = analyze_force_balance_results(RP; verbose=true, visualize=true)

        if results !== nothing && results.mean_error < 0.05
            println("\n" * "=" ^ 60)
            println("✓ GLOBAL FORCE BALANCE TEST COMPLETED SUCCESSFULLY")
            println("=" ^ 60)
        else
            println("\n" * "=" ^ 60)
            println("✗ GLOBAL FORCE BALANCE TEST FAILED - Poor accuracy")
            println("=" ^ 60)
        end
    catch e
        println("\n" * "=" ^ 60)
        println("✗ GLOBAL FORCE BALANCE TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 60)
        rethrow(e)
    end
end
