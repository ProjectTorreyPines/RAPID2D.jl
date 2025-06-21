#!/usr/bin/env julia
"""
Induced Current Test for RAPID2D.jl

Tests the effect of external coils on plasma motion by comparing:
1. Plasma motion without external coils (baseline)
2. Plasma motion with external coils that generate opposing induced currents

The external coils should slow down the plasma motion due to electromagnetic braking.
"""

using RAPID2D
using RAPID2D.Statistics
using Test
using Printf

# Environment variable controls
verbose = get(ENV, "RAPID_VERBOSE", "false") == "true"
visualize = get(ENV, "RAPID_VISUALIZE", "false") == "true"

if visualize
    using Plots
    using Dates
end

function run_induced_current_test(; verbose=false, visualize=false)
    if verbose
        println("=" ^ 60)
        println("RAPID2D.jl Induced Current Test")
        println("=" ^ 60)
    end

    # Run baseline case without coils
    if verbose
        println("Running baseline case (no external coils)...")
    end
    RP_base = run_baseline_case(verbose)
    baseline_results = analyze_induced_current_results(RP_base, "baseline")

    # Run test case with external coils
    if verbose
        println("Running test case (with external coils)...")
    end
    RP_with_coils = run_coil_case(verbose)
    coil_results = analyze_induced_current_results(RP_with_coils, "with_coils")

    # Compare results
    comparison = compare_results(baseline_results, coil_results, verbose, visualize)

    return (comparison, RP_base, RP_with_coils)
end

function run_baseline_case(verbose::Bool)
    """Run simulation without external coils as baseline"""
    config = create_induced_current_config(with_coils=false)
    RP = RAPID{Float64}(config)

    setup_induced_current_flags!(RP)
    initialize!(RP)
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)

    update_transport_quantities!(RP)

    if verbose
        println("  ✓ Baseline simulation initialized")
    end

    return run_simulation!(RP)
end

function run_coil_case(verbose::Bool)
    """Run simulation with external coils"""
    config = create_induced_current_config(with_coils=true)
    RP = RAPID{Float64}(config)

    setup_induced_current_flags!(RP)
    initialize!(RP)
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)
    setup_external_coils!(RP; verbose)


    RAPID2D.solve_Ampere_equation!(RP)
    combine_external_and_self_fields!(RP)
    update_transport_quantities!(RP)

    if verbose
        println("  ✓ Coil simulation initialized")
    end

    return run_simulation!(RP)
end

function create_induced_current_config(; with_coils=false)
    """Create configuration for induced current testing"""
    config = SimulationConfig{Float64}()

    # Grid parameters
    config.NR = 30
    config.NZ = 50

    # Physical parameters
    config.prefilled_gas_pressure = 1e-3  # Pa
    config.R0B0 = 3.0  # Tesla⋅meter

    # Time parameters - shorter simulation to focus on initial motion
    config.dt = 5e-6
    config.snap0D_Δt_s = 10e-6
    config.snap2D_Δt_s = 20e-6
    # config.t_end_s = 500e-6  # Shorter than force balance test
    config.t_end_s = 1e-3  # Shorter than force balance test

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_induced_current_flags!(RP::RAPID)
    """Set up simulation flags for induced current testing"""
    RP.flags = SimulationFlags{Float64}(
        Atomic_Collision = true,
        Te_evolve = false,
        Ti_evolve = false,
        src = false,
        convec = true,
        diffu = false,
        ud_evolve = true,
        Include_ud_convec_term = true,
        Include_ud_diffu_term = true,
        Include_ud_pressure_term = true,
        Include_Te_convec_term = true,
        update_ni_independently = false,
        Gas_evolve = false,
        Ampere = true,
        E_para_self_ES = false,
        negative_n_correction = true
    )

    # Additional flags
    RP.flags.mean_ExB = false
    RP.flags.turb_ExB_mixing = false
    RP.flags.E_para_self_ES = false
    RP.flags.Coulomb_Collision = true
    RP.flags.E_para_self_EM = true
    RP.flags.Ampere_Itor_threshold = 0.0
    RP.flags.FLF_nstep = 10
    RP.flags.Implicit = true
    RP.flags.Damp_Transp_outWall = true
    RP.flags.Global_JxB_Force = true
end

function setup_magnetic_field!(RP::RAPID; verbose::Bool=false)
    """Set up pure toroidal magnetic field"""
    # Zero poloidal field components
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
    E0 = 0.0  # V/m
    @. RP.fields.Eϕ = E0 * mean(RP.G.R1D) / RP.G.Jacob
    @. RP.fields.Eϕ_ext = RP.fields.Eϕ
    @. RP.fields.E_para_ext = RP.fields.Eϕ * RP.fields.bϕ

    if verbose
        println("  ✓ Magnetic field configuration set")
    end
end

function setup_plasma!(RP::RAPID; verbose::Bool=false)
    """Set up initial plasma density distribution"""
    # Plasma parameters - same as force balance test
    cenR = 1.3  # m
    cenZ = 0.0  # m
    radius = 0.2  # m
    n0 = 1e18  # m⁻³

    # Create Gaussian plasma profile
    ini_n = @. n0 * exp(-(((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) / (2 * radius^2)))

    # Apply spatial mask
    mask = @. sqrt((RP.G.R2D - cenR)^2 + (RP.G.Z2D - cenZ)^2) < radius
    ini_n .*= mask

    # Set plasma densities
    @. RP.plasma.ne = ini_n
    @. RP.plasma.ni = ini_n

    # Set temperatures
    fill!(RP.plasma.Te_eV, 10.0)  # eV
    fill!(RP.plasma.Ti_eV, 0.03)  # eV

    # Zero initial velocities
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ue_para, 1e6)
    fill!(RP.plasma.ui_para, 0.0)

    if verbose
        println("  ✓ Initial plasma configuration set")
    end
end

function setup_external_coils!(RP::RAPID; verbose::Bool=false)
    """Set up external coils near the outer wall to create opposing currents"""

    RP.coil_system = CoilSystem{Float64}()
    csys = RP.coil_system

    # Get wall boundaries
    wall_R_outer = maximum(RP.fitted_wall.R)
    wall_R_inner = minimum(RP.fitted_wall.R)

    # Place coils just outside the outer wall
    N_coils = 10
    coils_R = (wall_R_outer + 0.1)*ones(N_coils)  # 10 cm outside wall
    # coils_Z = collect(range(-0.3, 0.3, length=N_coils))
    coils_Z = collect(range(-1.0, 1.0, length=N_coils))
    # coils_Z = [-0.3, 0.0, 0.3]  # Three coils at different Z positions

    # Create external coils that will oppose plasma motion
    # When plasma moves outward and creates changing flux, these coils will induce currents

    # Coil specifications
    coil_resistivity = 1e4*csys.cu_resistivity
    coil_area = π * 0.05^2  # 5 cm radius coil area
    # coil_area = π * 0.01^2  # 5 cm radius coil area
    coil_resistance = 1e-3  # 1 mΩ resistance (low resistance for strong currents)
    coil_resistance = 1e-4  # 1 mΩ resistance (low resistance for strong currents)
    coil_resistance = 100.0  # 1 mΩ resistance (low resistance for strong currents)
    coil_self_inductance = 1e-6  # 1 μH self-inductance
    coil_self_inductance = 1e-8  # 1 μH self-inductance

    initial_current = 0.0  # Start with zero current

    # Create coils and add to system
    for i in 1:N_coils
        coil_name = "braking_coil_$(i)"
        coil_location = (r=coils_R[i], z=coils_Z[i])

        # Create passive coil (not externally powered, but can carry induced currents)
        braking_coil = Coil(
            coil_location,
            coil_area,
            calculate_coil_resistance(coil_area, coil_location.r, coil_resistivity),
            calculate_self_inductance(coil_area, coil_location.r, csys.μ0),
            false,  # not powered (passive)
            false,  # not controllable
            coil_name,
            nothing,  # no max voltage
            nothing,  # no max current
            initial_current,  # initial current
            0.0       # no external voltage
        )

        # Add coil to the system
        add_coil!(RP.coil_system, braking_coil)
    end

    initialize_coil_system!(RP)

    if verbose
        println("  ✓ External coils configured")
        println("    Number of coils: $(length(coils_Z))")
        println("    Coil Z positions: $(coils_Z) m")
        println("    Initial current per coil: $(initial_current) A")
        println("    Total coils in system: $(RP.coil_system.n_total)")
    end
end

function analyze_induced_current_results(RP::RAPID, case_name::String)
    """Analyze simulation results for induced current effects"""

    snaps0D = RP.diagnostics.snaps0D
    times = snaps0D.time_s
    ne_cen_R = snaps0D.ne_cen_R
    ne_cen_Z = snaps0D.ne_cen_Z
    avg_ueR = snaps0D.ueR
    avg_aR_by_JxB = snaps0D.aR_by_JxB
    total_ne = snaps0D.ne * RP.G.device_inVolume

    # Calculate motion characteristics
    initial_R = ne_cen_R[1]
    final_R = ne_cen_R[end]
    displacement_R = final_R - initial_R

    # Calculate average outward velocity and acceleration
    avg_velocity = length(times) > 1 ? displacement_R / (times[end] - times[1]) : 0.0
    avg_acceleration = length(avg_aR_by_JxB) > 0 ? mean(avg_aR_by_JxB) : 0.0

    # Time to reach 10% of maximum displacement
    target_displacement = 0.1 * displacement_R
    time_to_10pct = nothing
    for i in eachindex(ne_cen_R)
        if abs(ne_cen_R[i] - initial_R) >= abs(target_displacement)
            time_to_10pct = times[i]
            break
        end
    end

    return (; case_name, times, ne_cen_R, ne_cen_Z, avg_ueR, avg_aR_by_JxB,
             total_ne, displacement_R, avg_velocity, avg_acceleration,
             time_to_10pct, initial_R, final_R)
end

function compare_results(baseline, coil_case, verbose::Bool, visualize::Bool)
    """Compare baseline and coil case results"""

    if verbose
        println("\n" * "=" ^ 60)
        println("INDUCED CURRENT COMPARISON")
        println("=" ^ 60)
    end

    # Calculate relative changes
    velocity_reduction = (baseline.avg_velocity - coil_case.avg_velocity) / baseline.avg_velocity
    acceleration_reduction = (baseline.avg_acceleration - coil_case.avg_acceleration) / baseline.avg_acceleration
    displacement_reduction = (baseline.displacement_R - coil_case.displacement_R) / baseline.displacement_R

    # Time delay effect
    time_delay = nothing
    if baseline.time_to_10pct !== nothing && coil_case.time_to_10pct !== nothing
        time_delay = coil_case.time_to_10pct - baseline.time_to_10pct
    end

    if verbose
        println("Baseline (no coils):")
        println(@sprintf("  Final displacement: %.3f m", baseline.displacement_R))
        println(@sprintf("  Average velocity: %.1f m/s", baseline.avg_velocity))
        println(@sprintf("  Average acceleration: %.1f m/s²", baseline.avg_acceleration))
        if baseline.time_to_10pct !== nothing
            println(@sprintf("  Time to 10%% displacement: %.1f μs", baseline.time_to_10pct * 1e6))
        end

        println("\nWith external coils:")
        println(@sprintf("  Final displacement: %.3f m", coil_case.displacement_R))
        println(@sprintf("  Average velocity: %.1f m/s", coil_case.avg_velocity))
        println(@sprintf("  Average acceleration: %.1f m/s²", coil_case.avg_acceleration))
        if coil_case.time_to_10pct !== nothing
            println(@sprintf("  Time to 10%% displacement: %.1f μs", coil_case.time_to_10pct * 1e6))
        end

        println("\nCoil effects:")
        println(@sprintf("  Velocity reduction: %.1f%%", 100 * velocity_reduction))
        println(@sprintf("  Acceleration reduction: %.1f%%", 100 * acceleration_reduction))
        println(@sprintf("  Displacement reduction: %.1f%%", 100 * displacement_reduction))
        if time_delay !== nothing
            println(@sprintf("  Time delay: %.1f μs", time_delay * 1e6))
        end
    end

    # Create visualization
    if visualize
        create_comparison_plots(baseline, coil_case)
    end

    return (; baseline, coil_case, velocity_reduction, acceleration_reduction,
             displacement_reduction, time_delay)
end

function create_comparison_plots(baseline, coil_case)
    """Create comparison plots for baseline vs coil cases"""

    times_ms_base = baseline.times * 1e3
    times_ms_coil = coil_case.times * 1e3

    # 1. Centroid position comparison
    p1 = plot(times_ms_base, baseline.ne_cen_R,
              label="Baseline (no coils)", linewidth=2, color=:blue,
              xlabel="Time (ms)", ylabel="Radial Position (m)",
              title="Plasma Centroid Motion")
    plot!(p1, times_ms_coil, coil_case.ne_cen_R,
          label="With external coils", linewidth=2, color=:red)

    # 2. Velocity comparison
    p2 = plot(times_ms_base, baseline.avg_ueR,
              label="Baseline", linewidth=2, color=:blue,
              xlabel="Time (ms)", ylabel="Radial Velocity (m/s)",
              title="Plasma Velocity")
    plot!(p2, times_ms_coil, coil_case.avg_ueR,
          label="With coils", linewidth=2, color=:red)

    # 3. Acceleration comparison
    p3 = plot(times_ms_base, baseline.avg_aR_by_JxB,
              label="Baseline", linewidth=2, color=:blue,
              xlabel="Time (ms)", ylabel="Acceleration (m/s²)",
              title="JxB Acceleration")
    plot!(p3, times_ms_coil, coil_case.avg_aR_by_JxB,
          label="With coils", linewidth=2, color=:red)

    # 4. Phase space comparison
    if length(baseline.ne_cen_R) > 1 && length(coil_case.ne_cen_R) > 1
        p4 = plot(baseline.ne_cen_R, baseline.avg_ueR,
                  label="Baseline", linewidth=2, color=:blue,
                  xlabel="Radial Position (m)", ylabel="Radial Velocity (m/s)",
                  title="Phase Space (R vs vR)")
        plot!(p4, coil_case.ne_cen_R, coil_case.avg_ueR,
              label="With coils", linewidth=2, color=:red)
    else
        p4 = plot(title="Phase Space\n(Insufficient data)")
    end

    # Combined plot
    plot_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = joinpath(pwd(), "induced_current_test_$(timestamp).png")
    savefig(plot_combined, filename)

    println("\nVisualization:")
    println("  ✓ Comparison plot saved as: $(filename)")

    return plot_combined
end

@testset "RAPID2D.jl Induced Current Test" begin
    # Run the induced current test
    comparison, RP_baseline, RP_with_coils = run_induced_current_test(verbose=verbose, visualize=visualize)

    @test comparison !== nothing

    if comparison !== nothing
        baseline = comparison.baseline
        coil_case = comparison.coil_case

        # Basic data integrity tests
        @testset "Data Integrity" begin
            @test length(baseline.times) > 1
            @test length(coil_case.times) > 1
            @test length(baseline.ne_cen_R) == length(baseline.times)
            @test length(coil_case.ne_cen_R) == length(coil_case.times)
        end

        # Test that both cases show outward motion
        @testset "Plasma Motion" begin
            @test baseline.displacement_R > 0.01  # At least 1 cm outward motion
            @test coil_case.displacement_R > 0.01  # At least 1 cm outward motion
            @test baseline.avg_velocity > 0  # Outward velocity
            @test coil_case.avg_velocity > 0  # Outward velocity
        end

        # Test electromagnetic braking effect
        @testset "Electromagnetic Braking" begin
            # Coils should reduce plasma motion (electromagnetic braking)
            @test coil_case.avg_velocity < baseline.avg_velocity  # Slower with coils
            @test coil_case.displacement_R < baseline.displacement_R  # Less displacement

            # Velocity reduction should be significant (at least 5%)
            velocity_reduction = comparison.velocity_reduction
            @test velocity_reduction > 0.05  # At least 5% reduction
            @test velocity_reduction < 0.8   # But not too extreme (less than 80%)

            # Displacement reduction should be noticeable
            displacement_reduction = comparison.displacement_reduction
            @test displacement_reduction > 0.02  # At least 2% reduction
        end

        # Test timing effects
        @testset "Timing Effects" begin
            if comparison.time_delay !== nothing
                # External coils should delay plasma motion
                @test comparison.time_delay > 0  # Positive time delay
                @test comparison.time_delay < baseline.times[end]  # Reasonable delay
            end
        end

        # Test physical bounds
        @testset "Physical Bounds" begin
            # Both cases should stay within simulation domain
            @test all(baseline.ne_cen_R .>= baseline.initial_R - 0.1)
            @test all(coil_case.ne_cen_R .>= coil_case.initial_R - 0.1)

            # No NaN or Inf values
            @test all(isfinite.(baseline.ne_cen_R))
            @test all(isfinite.(coil_case.ne_cen_R))
            @test all(isfinite.(baseline.avg_ueR))
            @test all(isfinite.(coil_case.avg_ueR))
        end

        if verbose
            println("\nTest Summary:")
            println("  ✓ Electromagnetic braking detected")
            println(@sprintf("  ✓ Velocity reduction: %.1f%%", 100 * comparison.velocity_reduction))
            println(@sprintf("  ✓ Displacement reduction: %.1f%%", 100 * comparison.displacement_reduction))
            if comparison.time_delay !== nothing
                println(@sprintf("  ✓ Motion delay: %.1f μs", comparison.time_delay * 1e6))
            end
        end
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl Induced Current Test...")

    try
        comparison, RP_baseline, RP_with_coils = run_induced_current_test(verbose=verbose, visualize=visualize)

        if comparison !== nothing && comparison.velocity_reduction > 0.05
            println("\n" * "=" ^ 60)
            println("✓ INDUCED CURRENT TEST COMPLETED SUCCESSFULLY")
            println("✓ Electromagnetic braking effect confirmed")
            println("=" ^ 60)
        else
            println("\n" * "=" ^ 60)
            println("✗ INDUCED CURRENT TEST FAILED - Insufficient braking effect")
            println("=" ^ 60)
        end
    catch e
        println("\n" * "=" ^ 60)
        println("✗ INDUCED CURRENT TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 60)
        rethrow(e)
    end
end
