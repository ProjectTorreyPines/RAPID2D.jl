#!/usr/bin/env julia
"""
Global JxB Force Test for RAPID2D.jl

Tests global force balance functionality with plasma position control,
validating J×B force calculations and plasma centroid tracking.
Based on test_force_balance.m from RAPID-2D MATLAB version.
"""

using RAPID2D
using RAPID2D.Statistics
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
    """Create configuration for global force balance testing"""

    config = SimulationConfig{Float64}()

    # Grid parameters (smaller grid for faster testing)
    config.NR = 30
    config.NZ = 50

    # Physical parameters
    config.prefilled_gas_pressure = 1e-3  # Pa (from MATLAB: 1e-3)
    config.R0B0 = 3.0  # Tesla⋅meter

    # Time parameters
    config.dt = 2.5e-6
    config.snap0D_Δt_s = 5e-6
    config.snap2D_Δt_s = 50e-6
    config.t_end_s = 200e-6

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_force_balance_flags!(RP::RAPID)
    """Set up simulation flags for global force balance testing"""

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
        negative_n_correction = true,
        mean_ExB = false,
        turb_ExB_mixing = false,
        Coulomb_Collision = true,
        E_para_self_EM = true,
        Implicit = true,
        Damp_Transp_outWall = true,
        Global_JxB_Force = true  # KEY FEATURE for this test
    )

    # Additional numerical parameters
    RP.flags.Ampere_Itor_threshold = 0.0
    RP.flags.FLF_nstep = 10
end

function setup_magnetic_field!(RP::RAPID; verbose::Bool=false)
    """Set up pure toroidal magnetic field configuration"""

    # Zero poloidal field components (pure toroidal field)
    fill!(RP.fields.BR, 0.0)
    fill!(RP.fields.BZ, 0.0)
    fill!(RP.fields.BR_ext, 0.0)
    fill!(RP.fields.BZ_ext, 0.0)

    # Set Jacobian and toroidal field: Bφ = R₀B₀/R
    @. RP.G.Jacob = RP.G.R2D
    @. RP.fields.Bϕ = RP.config.R0B0 / RP.G.Jacob

    # Update total field and unit vectors
    @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
    @. RP.fields.Btot = abs(RP.fields.Bϕ)
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

    # Plasma parameters
    cenR = 1.3  # m
    cenZ = 0.1  # m
    radius = 0.2  # m
    n0 = 1e16  # m⁻³

    # Create Gaussian plasma profile
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
    """Analyze simulation results focusing on JxB force-driven plasma motion and wall interaction"""

    if verbose
        println("\n" * "=" ^ 60)
        println("GLOBAL JxB FORCE ANALYSIS")
        println("=" ^ 60)
    end

    # Extract time series data using getproperty overloading
    snaps0D = RP.diagnostics.snaps0D
    times = snaps0D.time_s
    I_tor = snaps0D.I_tor
    ne_cen_R = snaps0D.ne_cen_R
    ne_cen_Z = snaps0D.ne_cen_Z
    J_cen_R = snaps0D.J_cen_R
    J_cen_Z = snaps0D.J_cen_Z

    # JxB force and velocity data
    avg_ueR = snaps0D.ueR
    avg_ueZ = snaps0D.ueZ
    avg_aR_by_JxB = snaps0D.aR_by_JxB
    avg_aZ_by_JxB = snaps0D.aZ_by_JxB

    # Total plasma density
    total_ne = snaps0D.ne * RP.G.device_inVolume
    max_ne = snaps0D.ne_max

    # Initial conditions (for reference)
    initial_R = 1.3  # m (from setup_plasma!)
    initial_Z = 0.1  # m
    plasma_radius = 0.2  # m
    wall_R_inner = minimum(RP.fitted_wall.R)
    wall_R_outer = maximum(RP.fitted_wall.R)

    # Analyze plasma motion physics
    dt = length(times) > 1 ? times[2] - times[1] : 1e-6

    # 1. Plasma centroid motion analysis
    displacement_R = ne_cen_R .- initial_R
    displacement_Z = ne_cen_Z .- initial_Z

    # Calculate velocities from centroid positions using accurate numerical differentiation
    velocity_R_from_pos = Float64[]
    velocity_Z_from_pos = Float64[]

    if length(times) > 1
        for i in 1:length(times)
            if i == 1
                # Forward difference for first point
                if length(times) > 1
                    dt = times[2] - times[1]
                    vel_R = (ne_cen_R[2] - ne_cen_R[1]) / dt
                    vel_Z = (ne_cen_Z[2] - ne_cen_Z[1]) / dt
                else
                    vel_R = 0.0
                    vel_Z = 0.0
                end
            elseif i == length(times)
                # Backward difference for last point
                dt = times[end] - times[end-1]
                vel_R = (ne_cen_R[end] - ne_cen_R[end-1]) / dt
                vel_Z = (ne_cen_Z[end] - ne_cen_Z[end-1]) / dt
            else
                # Central difference for middle points (more accurate)
                dt = times[i+1] - times[i-1]
                vel_R = (ne_cen_R[i+1] - ne_cen_R[i-1]) / dt
                vel_Z = (ne_cen_Z[i+1] - ne_cen_Z[i-1]) / dt
            end

            push!(velocity_R_from_pos, vel_R)
            push!(velocity_Z_from_pos, vel_Z)
        end
    end

    # 2. Check consistency between acceleration, velocity, and position
    # Compare measured velocity (avg_ueR) with velocity from position derivative
    velocity_consistency_error = Float64[]
    acceleration_consistency_error = Float64[]

    # Velocity consistency: compare avg_ueR with numerical derivative of position
    for i in eachindex(velocity_R_from_pos)
        measured_vel = avg_ueR[i]
        computed_vel = velocity_R_from_pos[i]

        if abs(measured_vel) + abs(computed_vel) > 1e-10
            error = abs(measured_vel - computed_vel) / abs(measured_vel)
            push!(velocity_consistency_error, error)
        end
    end

    # Acceleration-velocity consistency: v(t) ≈ v(t-dt) + a(t)*dt
    if length(avg_aR_by_JxB) > 1 && length(avg_ueR) > 1
        for i in 2:min(length(avg_aR_by_JxB), length(avg_ueR))
            dt_local = times[i] - times[i-1]
            expected_vel = avg_ueR[i-1] + avg_aR_by_JxB[i] * dt_local
            actual_vel = avg_ueR[i]

            if abs(expected_vel) + abs(actual_vel) > 1e-10
                error = abs(expected_vel - actual_vel) / abs(actual_vel)
                push!(acceleration_consistency_error, error)
            end
        end
    end

    # 3. Identify wall collision time
    wall_collision_time = nothing
    wall_collision_idx = nothing
    density_drop_time = nothing
    density_drop_idx = nothing

    # Detect when plasma hits the wall (centroid reaches wall radius)
    for i in eachindex(ne_cen_R)
        if ne_cen_R[i] >= (wall_R_outer - plasma_radius) || ne_cen_R[i] <= (wall_R_inner + plasma_radius)
            wall_collision_time = times[i]
            wall_collision_idx = i
            break
        end
    end

    # Detect density drop (when total density drops to 95% of initial)
    initial_density = total_ne[1]
    for i in eachindex(total_ne)
        if total_ne[i] < 0.95 * initial_density
            density_drop_time = times[i]
            density_drop_idx = i
            break
        end
    end


    # 4. Calculate expected collision time with linear acceleration model
    expected_collision_time = nothing
    if length(avg_aR_by_JxB) > 5
        early_end_idx = min(length(avg_aR_by_JxB), wall_collision_idx !== nothing ? wall_collision_idx : length(avg_aR_by_JxB))
        early_indices = 2:min(10, early_end_idx)

        if length(early_indices) >= 3
            early_times = times[early_indices]
            early_accels = avg_aR_by_JxB[early_indices]
            n = length(early_times)

            # Linear regression: a(t) = a0 + k*t
            sum_t = sum(early_times)
            sum_t2 = sum(early_times.^2)
            sum_a = sum(early_accels)
            sum_at = sum(early_accels .* early_times)

            k = (n * sum_at - sum_t * sum_a) / (n * sum_t2 - sum_t^2)
            a0 = (sum_a - k * sum_t) / n

            distance_to_wall = (wall_R_outer - plasma_radius/2) - initial_R

            if k > 1e-10   # Linear acceleration model
                # Solve: k*t³ + 3*a0*t² - 6*distance = 0 (Newton's method)
                t_guess = sqrt(2 * distance_to_wall / max(mean(early_accels), 1e-10))

                for iter in 1:10
                    f = k * t_guess^3 + 3 * a0 * t_guess^2 - 6 * distance_to_wall
                    df = 3 * k * t_guess^2 + 6 * a0 * t_guess
                    if abs(df) < 1e-15 || abs(f) < 1e-12
                        break
                    end
                    t_new = t_guess - f / df
                    if abs(t_new - t_guess) < 1e-8 || t_new <= 0
                        break
                    end
                    t_guess = t_new
                end

                if t_guess > 0 && t_guess < 10 * times[end]
                    expected_collision_time = t_guess
                end
            elseif a0 > 0  # Constant acceleration fallback
                expected_collision_time = sqrt(2 * distance_to_wall / a0)
            end
        else
            # Simple constant acceleration estimate
            early_accel = mean(avg_aR_by_JxB[2:min(5, length(avg_aR_by_JxB))])
            distance_to_wall = (wall_R_outer - plasma_radius/2) - initial_R
            if early_accel > 0
                expected_collision_time = sqrt(2 * distance_to_wall / early_accel)
            end
        end
    end

    if verbose
        println("Plasma Motion Analysis:")
        println(@sprintf("  Initial position: R=%.2f m, Z=%.2f m", initial_R, initial_Z))
        println(@sprintf("  Final centroid: R=%.3f m, Z=%.3f m", ne_cen_R[end], ne_cen_Z[end]))
        println(@sprintf("  Max displacement: ΔR=%.3f m", maximum(abs.(displacement_R))))

        if length(avg_aR_by_JxB) > 5
            println(@sprintf("  Average early acceleration: %.1f m/s²", mean(avg_aR_by_JxB[2:min(5, length(avg_aR_by_JxB))])))
        end

        if !isempty(velocity_consistency_error)
            println(@sprintf("  Velocity consistency error: %.1f%%", 100*mean(velocity_consistency_error)))
        end

        if wall_collision_time !== nothing
            println(@sprintf("  Wall collision at: %.1f μs", wall_collision_time * 1e6))
        end

        if expected_collision_time !== nothing
            println(@sprintf("  Expected collision time: %.1f μs", expected_collision_time * 1e6))
        end

        println(@sprintf("  Density retention: %.1f%%", 100*total_ne[end]/initial_density))
    end

    # Create plots focusing on force balance aspects
    if visualize
        create_force_balance_plots(RP, times, ne_cen_R, ne_cen_Z, total_ne, avg_aR_by_JxB, wall_collision_time)
    end

    # Return analysis results
    return (; times, ne_cen_R, ne_cen_Z, displacement_R, displacement_Z, total_ne,
             avg_ueR, avg_aR_by_JxB, velocity_consistency_error, acceleration_consistency_error,
             wall_collision_time, density_drop_time, expected_collision_time, initial_density,
             final_density_fraction = total_ne[end] / initial_density,
             wall_R_outer, initial_R, plasma_radius, density_drop_idx)
end

function create_force_balance_plots(RP, times, ne_cen_R, ne_cen_Z, total_ne, avg_aR_by_JxB, wall_collision_time)
    """Create visualization plots for JxB force-driven plasma motion"""

    # Convert times to ms for plotting
    times_ms = times * 1e3
    wall_collision_ms = wall_collision_time !== nothing ? wall_collision_time * 1e3 : nothing

    # 1. Plasma centroid trajectory
    p1 = plot(times_ms, ne_cen_R,
              label="Density centroid R",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Radial Position (m)",
              title="Plasma Centroid Motion",
              margin=7Plots.mm)

    plot!(p1, times_ms, ne_cen_Z,
          label="Density centroid Z",
          linewidth=2)

    # Mark wall collision if detected
    if wall_collision_ms !== nothing
        vline!(p1, [wall_collision_ms],
               label="Wall collision",
               linestyle=:dash,
               color=:red)
    end

    # 2. Total density evolution
    p2 = plot(times_ms, total_ne,
              label="Total electron density",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Total ne (m⁻³)",
              title="Plasma Density Loss",
              yscale=:log10)

    if wall_collision_ms !== nothing
        vline!(p2, [wall_collision_ms],
               label="Wall collision",
               linestyle=:dash,
               color=:red)
    end

    # 3. JxB force and acceleration
    p3 = plot(times_ms, avg_aR_by_JxB,
              label="Radial acceleration",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Acceleration (m/s²)",
              title="JxB Force Acceleration")

    # 4. Phase space plot (position vs velocity if possible)
    if length(ne_cen_R) > 1
        velocity_R = diff(ne_cen_R) ./ diff(times)
        p4 = plot(ne_cen_R[1:end-1], velocity_R,
                  label="Trajectory",
                  linewidth=2,
                  xlabel="Radial Position (m)",
                  ylabel="Radial Velocity (m/s)",
                  title="Phase Space (R vs vR)")

        # Mark initial and final points
        scatter!(p4, [ne_cen_R[1]], [velocity_R[1]],
                label="Start", color=:green, markersize=6)
        if length(velocity_R) > 1
            scatter!(p4, [ne_cen_R[end-1]], [velocity_R[end]],
                    label="End", color=:red, markersize=6)
        end
    else
        p4 = plot(title="Phase Space\n(Insufficient data)")
    end

    # Combined plot
    plot_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = "JxB_force_test_$(timestamp).png"
    savefig(plot_combined, filename)

    println("\nVisualization:")
    println("  ✓ Plot saved as: $(filename)")

    # Create animation if possible
    try
        animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                    [:ne,:ue_para,:Jϕ,:E_para_tot,:ueR, :ueZ];
					wall=RP.fitted_wall,
                    filename="JxB_force_test_snaps2D_$(timestamp).mp4")
        println("  ✓ Animation saved as: JxB_force_test_snaps2D_$(timestamp).mp4")
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

    # Basic test assertions
    @test results !== nothing  # Should return valid results
    @test length(results.times) > 1  # Should have multiple time points
    @test length(results.ne_cen_R) == length(results.times)  # Consistent data arrays

    if results !== nothing
        # 1. Test plasma motion physics
        @testset "Plasma Motion Physics" begin
            # Plasma should move outward (away from initial position)
            max_displacement = maximum(abs.(results.displacement_R))
            @test max_displacement > 0.01  # Should move at least 1 cm

            # Final position should be farther from center than initial
            @test abs(results.ne_cen_R[end] - results.initial_R) > abs(results.displacement_R[1])

            # Plasma motion should be primarily radial (outward) for this test case
            radial_motion = abs(results.ne_cen_R[end] - results.ne_cen_R[1])
            vertical_motion = abs(results.ne_cen_Z[end] - results.ne_cen_Z[1])
            @test radial_motion > 0.5 * vertical_motion  # Predominantly radial motion
        end

        # 2. Test force and kinematics consistency
        @testset "Force-Kinematics Consistency" begin
            # Test velocity consistency (measured vs computed from position)
            if !isempty(results.velocity_consistency_error)
                mean_velocity_error = mean(results.velocity_consistency_error[2:results.density_drop_idx])
                @test mean_velocity_error < 0.2  # Less than 20% error in velocity consistency
            end

            # Test acceleration-velocity consistency
            if !isempty(results.acceleration_consistency_error)
                mean_accel_error = mean(results.acceleration_consistency_error[2:results.density_drop_idx])
                @test mean_accel_error < 0.3  # Less than 30% error (more tolerant for integration)
            end

            # Early phase should show positive outward acceleration
            early_accel = results.avg_aR_by_JxB[2:min(5, length(results.avg_aR_by_JxB))]
            @test mean(early_accel) > 0  # Outward acceleration
        end

        # 3. Test wall interaction and density loss
        @testset "Wall Interaction" begin
            # Density should decrease over time due to wall losses
            @test results.total_ne[end] < results.total_ne[1]  # Density decreases

            # Significant density loss should occur (plasma hits wall)
            @test results.final_density_fraction < 0.8  # Less than 80% density remaining

            # Wall collision should occur within reasonable time
            if results.wall_collision_time !== nothing
                @test results.wall_collision_time < maximum(results.times)  # Collision before end
                @test results.wall_collision_time > 0  # Positive collision time
            end

            # Density drop should correlate with wall collision
            if results.density_drop_time !== nothing && results.wall_collision_time !== nothing
                time_diff = abs(results.density_drop_time - results.wall_collision_time)
                simulation_duration = maximum(results.times) - minimum(results.times)
                @test time_diff < 0.2 * simulation_duration  # Within 20% of simulation time
            end
        end

        # 4. Test expected collision timing
        @testset "Collision Timing" begin
            if results.expected_collision_time !== nothing && results.wall_collision_time !== nothing
                # Predicted and actual collision times should be reasonably close
                timing_error = abs(results.expected_collision_time - results.wall_collision_time) / results.expected_collision_time
                @test timing_error < 0.5  # Within 50% of expected time (ballistic estimate)
            end
        end

        # 5. Test physical bounds and sanity checks
        @testset "Physical Bounds" begin
            # Centroid should stay within simulation domain
            @test all(results.ne_cen_R .>= minimum(RP.G.R1D))
            @test all(results.ne_cen_R .<= maximum(RP.G.R1D))
            @test all(results.ne_cen_Z .>= minimum(RP.G.Z1D))
            @test all(results.ne_cen_Z .<= maximum(RP.G.Z1D))

            # Density should remain positive
            @test all(results.total_ne .>= 0)

            # No NaN or Inf values in key quantities
            @test all(isfinite.(results.ne_cen_R))
            @test all(isfinite.(results.ne_cen_Z))
            @test all(isfinite.(results.total_ne))
        end

        # 6. Test simulation quality indicators
        @testset "Simulation Quality" begin
            # Should maintain reasonable plasma density for some time
            initial_phase_end = min(length(results.total_ne), div(length(results.total_ne), 3))
            early_density_retention = minimum(results.total_ne[1:initial_phase_end]) / results.total_ne[1]
            @test early_density_retention > 0.5  # Keep >50% density in early phase

            # Motion should be smooth (no large jumps in centroid position)
            if length(results.ne_cen_R) > 2
                position_jumps = abs.(diff(results.ne_cen_R))
                max_jump = maximum(position_jumps)
                average_motion = abs(results.ne_cen_R[end] - results.ne_cen_R[1]) / length(results.ne_cen_R)
                @test max_jump < 10 * average_motion  # No jumps >10x average step
            end
        end

        if verbose
            println("\nTest Summary:")
            println("  ✓ Max displacement: $(maximum(abs.(results.displacement_R))) m")
            println("  ✓ Density retained: $(round(100*results.final_density_fraction, digits=1))%")
            if results.wall_collision_time !== nothing
                println("  ✓ Wall collision at $(round(results.wall_collision_time*1e6, digits=1)) μs")
            end
            if !isempty(results.velocity_consistency_error)
                println("  ✓ Velocity error: $(round(100*mean(results.velocity_consistency_error), digits=1))%")
            end
        end
    end
end



# Main execution (when run directly as a script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl Global JxB Force Test...")

    try
        RP = run_global_force_test(verbose=true, visualize=true)
        results = analyze_force_balance_results(RP; verbose=true, visualize=true)

        println("\n" * "=" ^ 60)
        println("✓ GLOBAL FORCE BALANCE TEST COMPLETED")
        println("=" ^ 60)
    catch e
        println("\n" * "=" ^ 60)
        println("✗ GLOBAL FORCE BALANCE TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 60)
        rethrow(e)
    end
end
