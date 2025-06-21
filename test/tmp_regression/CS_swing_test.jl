#!/usr/bin/env julia
"""
CS Swing Test for RAPID2D.jl

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

FT = Float64

function run_CS_swing(verbose::Bool=false)
    """Run simulation with external coils"""
    config = create_config()
    RP = RAPID{FT}(config)

    setup_flags!(RP)
    initialize!(RP)
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)
    setup_external_coils!(RP; verbose)

    RAPID2D.solve_Ampere_equation!(RP; update_Eϕ_self=false)

    combine_external_and_self_fields!(RP)
    update_transport_quantities!(RP)

    if verbose
        println("  ✓ Coil simulation initialized")
    end

	# Ip_controller = create_controller(200.0, 1e-3,RP.coil_system.coils; Kp=-0.1, Ti=1e-2, control_type="current")
	# Ip_controller = create_controller(200.0, 1e-3,RP.coil_system.coils; Kp=-0.05, Ti=1e-3, Td=0.5e-3, control_type="current")
	# Ip_controller = create_controller(1e3, 1e-3,RP.coil_system.coils; Kp=-0.05, Ti=30e-3, Td=10e-3, control_type="current", N=2)

	# # Ip_controller = create_controller(1e3, 1e-3,RP.coil_system.coils; Kp=-0.05, Ti=Inf, Td=10e-3, control_type="current", N=2)
	# Ip_controller = create_controller(1e3, 1e-3,RP.coil_system.coils; Kp=-0.2, Ti=Inf, Td=0.1e-3, control_type="current", N=2)
	# Ip_controller = create_controller(1e3, 1e-3,RP.coil_system.coils; Kp=-0.10, Ti=20e-3, Td=0.05e-3, control_type="current", N=100, Ts=1e-10)

	Ip_controller = create_controller(1e3, 1e-3,RP.coil_system.coils; control_type="current",
										Kp=-0.05, Ti=0.05e-3, Td=0.0,  N=0.1, Ts=1e-6)

    return run_simulation!(RP; controller=Ip_controller)
end

function create_config()
    """Create configuration for CS Swing testing"""
    config = SimulationConfig{FT}()

    # Grid parameters
    config.NR = 30
    config.NZ = 50

    # Physical parameters
    config.prefilled_gas_pressure = 1e-3  # Pa
    config.R0B0 = 3.0  # Tesla⋅meter

    # Time parameters - shorter simulation to focus on initial motion
    config.dt = 5e-6
    config.snap0D_Δt_s = 10e-6
    config.snap2D_Δt_s = 100e-6
    # config.t_end_s = 500e-6  # Shorter than force balance test
    config.t_end_s = 1e-3  # Shorter than force balance test

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_flags!(RP::RAPID)
    """Set up simulation flags for CS Swing testing"""
    RP.flags = SimulationFlags{FT}(
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
    RP.flags.Global_JxB_Force = false
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
    Eϕ = @. E0 * mean(RP.G.R1D) / RP.G.Jacob
    RP.fields.Eϕ_ext .= Eϕ
    RP.fields.LV_ext .= Eϕ .* (2 * π * RP.G.R2D)
    # Parallel component of E
    RP.fields.E_para_ext .= Eϕ .* (RP.fields.Bϕ ./ RP.fields.Btot)

    if verbose
        println("  ✓ Magnetic field configuration set")
    end
end

function setup_plasma!(RP::RAPID; verbose::Bool=false)
    """Set up initial plasma density distribution"""
    # Plasma parameters - same as force balance test
    cenR = 1.5  # m
    cenZ = 0.0  # m
    radius = 0.4  # m
    n0 = 1e18  # m⁻³
    # n0 = 1e17  # m⁻³
    # n0 = 1e10  # m⁻³

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
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)

    if verbose
        println("  ✓ Initial plasma configuration set")
    end
end

function setup_external_coils!(RP::RAPID; verbose::Bool=false)
    """Set up external coils near the outer wall to create opposing currents"""
    RP.coil_system = CoilSystem{FT}()
    csys = RP.coil_system

    # Get wall boundaries
    wall_R_inner = minimum(RP.fitted_wall.R)

    # Place coils just outside the outer wall
    N_coils = 10
    coils_r = (wall_R_inner - 0.2)*ones(N_coils)  # 10 cm outside wall
    coils_z = collect(range(-1.0, 1.0, length=N_coils))

    # Coil specifications
    coil_resistivity = csys.cu_resistivity
    coil_area = π * 0.05^2  # 5 cm radius coil area

    initial_current = 0.0  # Start with zero current


    # Create coils and add to system
    for i in 1:N_coils
        coil_name = "CS_$(i)"
        coil_location = (r=coils_r[i], z=coils_z[i])

        CS_coil = Coil{FT}(
                location=coil_location,
                area=coil_area,
                resistance=calculate_coil_resistance(coil_area, coil_location.r, coil_resistivity),
                self_inductance=calculate_self_inductance(coil_area, coil_location.r, csys.μ0),
                is_powered=true,
                name="CS_$i",
                # current=0.0,
                # voltage_ext= t -> (t < 0.25e-3 ? 1e4*t*4 : 10.0)  # Step function voltage
                voltage_ext= t -> (t < 0.2e-3 ? 10.0 : max(-10.0, 10.0 - 10*( (t-0.2e-3)/0.5e-3)) ),  # Step function voltage
				current = 10.0/calculate_coil_resistance(coil_area, coil_location.r, coil_resistivity)
                # voltage_ext=-10.0
                # voltage_ext= t -> 0.0  # Step function voltage
                # voltage_ext= t -> -10.0  # Step function voltage
        )

        # Add coil to the system
        add_coil!(RP.coil_system, CS_coil)
    end

    initialize_coil_system!(RP)
    if verbose
        println("  ✓ External coils configured")
        println("    Number of coils: $(length(coils_z))")
        println("    Coil Z positions: $(coils_z) m")
        println("    Initial current per coil: $(initial_current) A")
        println("    Total coils in system: $(RP.coil_system.n_total)")
    end
end



# @testset "RAPID2D.jl CS Swing Test" begin
#     # Run the CS Swing test
#     comparison, RP_baseline, RP_with_coils = run_induced_current_test(verbose=verbose, visualize=visualize)

#     @test comparison !== nothing

#     if comparison !== nothing
#         baseline = comparison.baseline
#         coil_case = comparison.coil_case

#         # Basic data integrity tests
#         @testset "Data Integrity" begin
#             @test length(baseline.times) > 1
#             @test length(coil_case.times) > 1
#             @test length(baseline.ne_cen_R) == length(baseline.times)
#             @test length(coil_case.ne_cen_R) == length(coil_case.times)
#         end

#         # Test that both cases show outward motion
#         @testset "Plasma Motion" begin
#             @test baseline.displacement_R > 0.01  # At least 1 cm outward motion
#             @test coil_case.displacement_R > 0.01  # At least 1 cm outward motion
#             @test baseline.avg_velocity > 0  # Outward velocity
#             @test coil_case.avg_velocity > 0  # Outward velocity
#         end

#         # Test electromagnetic braking effect
#         @testset "Electromagnetic Braking" begin
#             # Coils should reduce plasma motion (electromagnetic braking)
#             @test coil_case.avg_velocity < baseline.avg_velocity  # Slower with coils
#             @test coil_case.displacement_R < baseline.displacement_R  # Less displacement

#             # Velocity reduction should be significant (at least 5%)
#             velocity_reduction = comparison.velocity_reduction
#             @test velocity_reduction > 0.05  # At least 5% reduction
#             @test velocity_reduction < 0.8   # But not too extreme (less than 80%)

#             # Displacement reduction should be noticeable
#             displacement_reduction = comparison.displacement_reduction
#             @test displacement_reduction > 0.02  # At least 2% reduction
#         end

#         # Test timing effects
#         @testset "Timing Effects" begin
#             if comparison.time_delay !== nothing
#                 # External coils should delay plasma motion
#                 @test comparison.time_delay > 0  # Positive time delay
#                 @test comparison.time_delay < baseline.times[end]  # Reasonable delay
#             end
#         end

#         # Test physical bounds
#         @testset "Physical Bounds" begin
#             # Both cases should stay within simulation domain
#             @test all(baseline.ne_cen_R .>= baseline.initial_R - 0.1)
#             @test all(coil_case.ne_cen_R .>= coil_case.initial_R - 0.1)

#             # No NaN or Inf values
#             @test all(isfinite.(baseline.ne_cen_R))
#             @test all(isfinite.(coil_case.ne_cen_R))
#             @test all(isfinite.(baseline.avg_ueR))
#             @test all(isfinite.(coil_case.avg_ueR))
#         end

#         if verbose
#             println("\nTest Summary:")
#             println("  ✓ Electromagnetic braking detected")
#             println(@sprintf("  ✓ Velocity reduction: %.1f%%", 100 * comparison.velocity_reduction))
#             println(@sprintf("  ✓ Displacement reduction: %.1f%%", 100 * comparison.displacement_reduction))
#             if comparison.time_delay !== nothing
#                 println(@sprintf("  ✓ Motion delay: %.1f μs", comparison.time_delay * 1e6))
#             end
#         end
#     end
# end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl CS Swing Test...")

    try
        RP = run_CS_swing(verbose=verbose, visualize=visualize)

        if comparison !== nothing && comparison.velocity_reduction > 0.05
            println("\n" * "=" ^ 60)
            println("✓ CS Swing TEST COMPLETED SUCCESSFULLY")
            println("✓ Electromagnetic braking effect confirmed")
            println("=" ^ 60)
        else
            println("\n" * "=" ^ 60)
            println("✗ CS Swing TEST FAILED - Insufficient braking effect")
            println("=" ^ 60)
        end
    catch e
        println("\n" * "=" ^ 60)
        println("✗ CS Swing TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 60)
        rethrow(e)
    end
end
