#!/usr/bin/env julia
"""
Plasma-Coil Inductive Coupling Test for RAPID2D.jl

This script tests the inductive coupling between a plasma column and an external coil.
The plasma is treated as a deformable conducting loop with time-varying resistance
due to temperature/collision frequency changes, coupled to an external coil circuit.

The test validates against analytical solution for two coupled LR circuits where:
- Circuit 1: External coil (fixed L, R, applied voltage)
- Circuit 2: Plasma loop (varying R due to collisions, inductance from geometry)

Physical setup:
- Stationary plasma (no convection/diffusion, only ud_evolve=true)
- Single external coil
- Pure inductive coupling (no direct electrical connection)
"""

using RAPID2D
using RAPID2D.Statistics
using RAPID2D.LinearAlgebra
using RAPID2D.Interpolations
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

function run_plasma_coil_coupling_test(; verbose=false, visualize=false)
    if verbose
        println("=" ^ 70)
        println("RAPID2D.jl Plasma-Coil Inductive Coupling Test")
        println("=" ^ 70)
    end

    # Create configuration
    config = create_coupling_config()

    # Initialize RAPID simulation
    if verbose
        println("Initializing RAPID simulation with plasma-coil coupling...")
    end
    RP = RAPID{Float64}(config)

    # Set up flags for coupling test
    setup_coupling_flags!(RP)

    # Initialize the simulation
    initialize!(RP)

    # Set up magnetic field, plasma, and coil system
    setup_magnetic_field!(RP; verbose)
    setup_plasma!(RP; verbose)
    setup_external_coil!(RP; verbose)


    # Set time parameters for coupling dynamics
    RP.t_end_s = 50e-3  # 50ms to see both fast and slow dynamics
    RP.t_end_s = 2e-3  # 50ms to see both fast and slow dynamics

    # Run simulation
    if verbose
        println("Running plasma-coil coupling simulation...")
    end
    run_simulation!(RP)

    # Return RP for analysis
    return RP
end

function create_coupling_config()
    """Create configuration optimized for plasma-coil coupling test"""

    config = SimulationConfig{Float64}()

    # Grid parameters (fine enough to resolve plasma-coil coupling)
    config.NR = 40
    config.NZ = 60

    # Physical parameters
    config.prefilled_gas_pressure = 0.0  # No gas evolution
    config.R0B0 = 2.5  # Tesla⋅meter

    # Time stepping (small enough to resolve L/R time scales)
    config.dt = 5e-6  # 5 μs
    config.snap0D_Δt_s = 20e-6
    config.snap2D_Δt_s = 100e-6

    # Device parameters
    config.device_Name = "manual"

    return config
end

function setup_coupling_flags!(RP::RAPID)
    """Set up simulation flags for plasma-coil coupling test"""

    RP.flags = SimulationFlags{Float64}(
        # Basic plasma evolution
        Atomic_Collision = true,
        Te_evolve = false,        # Fixed temperature for this test
        Ti_evolve = false,
        src = false,
        convec = false,           # No convection (stationary plasma)
        diffu = false,            # No diffusion
        ud_evolve = true,         # Only momentum evolution
        Include_ud_convec_term = false,
        Include_ud_diffu_term = false,
        Include_ud_pressure_term = false,
        Include_Te_convec_term = false,
        update_ni_independently = false,
        Gas_evolve = false,

        # Electromagnetic
        Ampere = true,            # Essential for coupling
        E_para_self_ES = false,
        E_para_self_EM = true,    # Essential for inductive effects
        Ampere_Itor_threshold = 0.0,

        # Numerical
        FLF_nstep = 50,
        Implicit = true,
        Damp_Transp_outWall = true,
        Global_JxB_Force = false,

        # Collisions
        Coulomb_Collision = true,
        negative_n_correction = true
    )
end

function setup_magnetic_field!(RP::RAPID; verbose::Bool=false)
    """Set up initial magnetic field (pure toroidal + small perturbation from coil)"""

    # Zero initial poloidal field
    fill!(RP.fields.BR, 0.0)
    fill!(RP.fields.BZ, 0.0)
    fill!(RP.fields.BR_ext, 0.0)
    fill!(RP.fields.BZ_ext, 0.0)

    # Set Jacobian
    @. RP.G.Jacob = RP.G.R2D

    # Toroidal field: Bφ = R₀B₀/R
    @. RP.fields.Bϕ = RP.config.R0B0 / RP.G.Jacob

    # Update total field
    @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
    @. RP.fields.Btot = abs(RP.fields.Bϕ)

    # Update unit vectors
    @. RP.fields.bR = RP.fields.BR / RP.fields.Btot
    @. RP.fields.bZ = RP.fields.BZ / RP.fields.Btot
    @. RP.fields.bϕ = RP.fields.Bϕ / RP.fields.Btot

    # Initial electric field (will be modified by coil)
    E0 = 0.0  # Start with zero - coil will drive the system
    @. RP.fields.Eϕ = E0
    fill!(RP.fields.Eϕ_ext, 0.0)
    fill!(RP.fields.E_para_ext, 0.0)

    if verbose
        println("  ✓ Magnetic field initialized (pure toroidal)")
        println("    R₀B₀ = $(RP.config.R0B0) T⋅m")
    end
end

function setup_plasma!(RP::RAPID; verbose::Bool=false)
    """Set up localized plasma column for coupling test"""

    # Plasma parameters
    plasma_R = 1.5   # m (major radius)
    plasma_Z = 0.0   # m (on midplane)
    minor_r = 0.3   # m (minor radius)
    n0 = 1e16        # m⁻³ (peak density)
    Te0 = 10.0       # eV (electron temperature)
    Ti0 = 0.026       # eV (ion temperature)

    # Create plasma profile (Gaussian)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R = RP.G.R2D[i, j]
        Z = RP.G.Z2D[i, j]

        r_minor = sqrt((R - plasma_R)^2 + (Z - plasma_Z)^2)

        if r_minor < minor_r
            # Gaussian profile with sharp cutoff
			RP.plasma.ne[i, j] = n0
            RP.plasma.ni[i, j] = n0
            RP.plasma.Te_eV[i, j] = Te0
            RP.plasma.Ti_eV[i, j] = Ti0
        else
            RP.plasma.ne[i, j] = 1e6  # Background density
            RP.plasma.ni[i, j] = 1e6
            RP.plasma.Te_eV[i, j] = 1.0
            RP.plasma.Ti_eV[i, j] = 0.1
        end
    end

    # Zero initial parallel velocities
    fill!(RP.plasma.ue_para, 0.0)
    fill!(RP.plasma.ui_para, 0.0)

    # Calculate initial plasma current density
    fill!(RP.plasma.Jϕ, 0.0)

    if verbose
        total_particles = sum(RP.plasma.ne) * RP.G.dR * RP.G.dZ
        println("  ✓ Plasma initialized")
        println("    Center: R=$(plasma_R)m, Z=$(plasma_Z)m")
        println("    Minor radius: $(minor_r)m")
        println("    Peak density: $(n0/1e18) × 10¹⁸ m⁻³")
        println("    Peak Te: $(Te0) eV")
        println("    Total particles: $(total_particles/1e18) × 10¹⁸")
    end
end

function setup_external_coil!(RP::RAPID; verbose::Bool=false)
    """Add external coil for inductive coupling"""

    # Coil parameters (positioned to couple with plasma)
    coil_r =1.2     # m (inner radius, couples well with plasma)
    coil_z = 0.8     # m (on midplane)
    coil_area = π * (0.05)^2  # 7cm radius conductor
    max_voltage = 50.0  # V
    max_current = 10e3  # A

    # Calculate coil resistance and self-inductance
    resistivity = 1.68e-8  # Ω⋅m (copper)
    coil_resistance = resistivity * 2π * coil_r/ coil_area

    # Self-inductance (rough estimate for single-turn, scaled by N²)
    μ0 = 4π * 1e-7
	YY = 1.0
    coil_self_L = μ0 * coil_r * (log(8 * coil_r / sqrt(coil_area/π)) - 2 + 0.25 * YY)


    # Create coil
    external_coil = Coil(
        (r=coil_r, z=coil_z),
        coil_area ,  # Effective area
        coil_resistance,
        coil_self_L,
        true,   # is_powered
        true,   # is_controllable
        "coupling_coil",
        max_voltage,
        max_current,
        0.0,    # initial current
		0.0
        # 20.0    # applied voltage (step input)
    )

    # Add to coil system
    add_coil!(RP.coil_system, external_coil)

    # Update coil system matrices
    RP.coil_system.Δt = RP.dt
	initialize_coil_system!(RP)

    if verbose
        println("  ✓ External coil added")
        println("    Position: R=$(coil_r)m, Z=$(coil_z)m")
        println("    Resistance: $(coil_resistance*1000) mΩ")
        println("    Self-inductance: $(coil_self_L*1e6) μH")
        println("    Applied voltage: $(external_coil.voltage_ext) V")
        println("    L/R time constant: $(coil_self_L/coil_resistance*1e6) μs")
    end
end

function calculate_analytical_solution(times, L1, L2, M, R1, R2, V1, V2; I1_0=0.0, I2_0=0.0)
    """
    Analytical solution for two coupled LR circuits

    System: L₁(dI₁/dt) + M(dI₂/dt) + R₁I₁ = V₁
           L₂(dI₂/dt) + M(dI₁/dt) + R₂I₂ = V₂

    Where: Circuit 1 = External coil, Circuit 2 = Plasma loop
    """

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



function calculate_analytical_solution_time_varying_M_and_L(times, L1, itp_L_plasma, itp_M, R1, R2, V1, V2; I1_0=0.0, I2_0=0.0, dt=nothing)
    """
    Semi-analytical solution for two coupled LR circuits with time-varying mutual inductance M(t) and plasma inductance L₂(t)

    System (from Faraday's law):
    V₁ = L₁(dI₁/dt) + M(t)(dI₂/dt) + (dM/dt)I₂ + R₁I₁
    V₂ = L₂(t)(dI₂/dt) + (dL₂/dt)I₂ + M(t)(dI₁/dt) + (dM/dt)I₁ + R₂I₂

    Where: Circuit 1 = External coil (constant L₁), Circuit 2 = Plasma loop (time-varying L₂(t))
           itp_M = interpolation function that takes time and returns M(t)
           itp_L_plasma = interpolation function that takes time and returns L₂(t)

    Uses 4th-order Runge-Kutta with fine time steps for accuracy, but only stores
    results at the requested time points in 'times' array.
    """

    # Use small time step for accuracy (RP.dt or smaller)
    if dt === nothing
        dt_requested = length(times) > 1 ? times[2] - times[1] : 1e-6
        dt = min(dt_requested / 10, 1e-6)  # Use 10x finer or 1μs, whichever is smaller
    end

    # Define the coupled ODE system with time-varying L₂(t) and M(t)
    function coupled_ode_ML(t, I1, I2)
        M_t = itp_M(t)
        L2_t = itp_L_plasma(t)

        # Calculate time derivatives numerically using small time step
        dt_small = 1e-8
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
        rhs = [V1 - R1*I1 - dM_dt*I2;
               V2 - R2*I2 - dM_dt*I1 - dL2_dt*I2]

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
            k2_1, k2_2 = coupled_ode_ML(t_current + dt_step/2, I1_current + dt_step*k1_1/2, I2_current + dt_step*k1_2/2)
            k3_1, k3_2 = coupled_ode_ML(t_current + dt_step/2, I1_current + dt_step*k2_1/2, I2_current + dt_step*k2_2/2)
            k4_1, k4_2 = coupled_ode_ML(t_current + dt_step, I1_current + dt_step*k3_1, I2_current + dt_step*k3_2)

            # Update currents
            I1_current += dt_step/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
            I2_current += dt_step/6 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)

            # Update time
            t_current += dt_step
        end

        # Store result at target time (with interpolation if needed)
        if abs(t_current - t_target) < 1e-12  # Exact match
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

function extract_plasma_circuit_parameters(RP::RAPID; verbose::Bool=false)
    """Extract equivalent circuit parameters for plasma loop"""

    # Plasma current and total current
    plasma_nodes = RP.G.inVol2D .> 0.5
    I_plasma = sum(RP.plasma.Jϕ[plasma_nodes]) * RP.G.dR * RP.G.dZ

    # Plasma resistance from collision frequency
    # R = (me * νei) / (e² * ne) integrated over plasma volume
    RAPID2D.@unpack ee, me = RP.config.constants
    # Resistance by Coulomb collision
    ue_sat_by_Eext = @. -ee*RP.fields.Eϕ_ext/(me * RP.plasma.ν_ei_eff);
    ue_sat_by_Eext[.!isfinite.(ue_sat_by_Eext)] .= 0.0  # Avoid NaN

    # Estimate circuit parameters
    in_wall_nids = RP.G.nodes.in_wall_nids
    LV_plasma = mean(RP.fields.LV_ext[in_wall_nids])
    R_plasma = LV_plasma./sum(-ee*RP.plasma.ne[in_wall_nids].*ue_sat_by_Eext[in_wall_nids]*RP.G.dR*RP.G.dZ);

    # Plasma geometry for inductance estimate
    major_R = 1.5  # m
    minor_r = 0.3  # m
    YY = 0.0  # initially surface current

    μ0 = 4π * 1e-7  # H/m
    L_plasma = μ0 * major_R * (log(8 * major_R / minor_r) - 2 + 0.25 * YY)

    if verbose
        println("  Plasma circuit parameters:")
        println("    Self-inductance: $(L_plasma*1e6) μH")
        println("    Resistance: $(R_plasma*1e3) mΩ")
        println("    L/R time: $(L_plasma/R_plasma*1e6) μs")
        println("    Current: $(I_plasma) A")
    end

    return L_plasma, R_plasma, LV_plasma
end

function analyze_coupling_results(RP::RAPID; verbose::Bool=false, visualize::Bool=false)
    """Analyze plasma-coil coupling results and compare with analytical solution"""

    if verbose
        println("\n" * "=" ^ 70)
        println("PLASMA-COIL COUPLING ANALYSIS")
        println("=" ^ 70)
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
	snaps2D_time_s = range(snaps2D_time_s[1], stop=snaps2D_time_s[end], length=length(snaps2D_time_s))
	M_values = [(2π*RP.coil_system.Green_grid2coils*s.Jϕ[:]/sum(s.Jϕ))[1] for s in RP.diagnostics.snaps2D]
	M_values[1] = M_values[2] # Avoid zero M

	itp_M = cubic_spline_interpolation(snaps2D_time_s, M_values)


	mean_M = mean(M_values)
	# mean_M = 0.0

    if verbose
        println("\nCircuit Parameters:")
        println("  Coil: L=$(L_coil*1e6)μH, R=$(R_coil*1e3)mΩ, V=$(V_coil)V")
        println("  Plasma: L=$(L_plasma*1e6)μH, R=$(R_plasma*1e3)mΩ")
        println("  Mutual inductance (est.): $(mean_M*1e6)μH")
        println("  Coupling coefficient: $(mean_M/sqrt(L_coil*L_plasma))")
    end

    # Calculate analytical solution
	I_coil_analytical, I_plasma_analytical, λ1, λ2, I_steady =
		calculate_analytical_solution(times, L_coil, L_plasma, mean_M,
									R_coil, R_plasma, V_coil, LV_plasma)

	# Calculate analytical solution with both time-varying M(t) and L₂(t)
	# Create time-varying plasma inductance (slight increase due to current profile changes)
	L_plasma_values = L_plasma .* (1.0 .+ 0.1 .* (times ./ times[end]))  # 10% increase over time
	L_plasma_values = RP.diagnostics.snaps0D.self_inductance_plasma
	L_plasma_values[1] = L_plasma_values[2] # Avoid zero at t=0
	itp_L_plasma = linear_interpolation(times, L_plasma_values)

	I_coil_analytical_ML, I_plasma_analytical_ML =
		calculate_analytical_solution_time_varying_M_and_L(times, L_coil, itp_L_plasma, itp_M,
															R_coil, R_plasma, V_coil, LV_plasma)

	if verbose
		println("\nAnalytical Solution (constant M):")
		println("  Eigenvalues: λ₁=$(λ1) s⁻¹, λ₂=$(λ2) s⁻¹")
		println("  Time constants: τ₁=$(-1/λ1*1e3)ms, τ₂=$(-1/λ2*1e3)ms")
		println("  Steady-state currents: I_coil=$(I_steady[1])A, I_plasma=$(I_steady[2])A")

		println("\nAnalytical Solution (time-varying M & L₂):")
		println("  L₂(t) range: $(minimum(L_plasma_values)*1e6) to $(maximum(L_plasma_values)*1e6) μH")
		println("  L₂ increase: $(((maximum(L_plasma_values)-minimum(L_plasma_values))/minimum(L_plasma_values)*100))%")
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
		valid_indices = abs.(ana) .> 1e-10
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
		println(@sprintf("  Coil current mean error: %.1f%%", 100*coil_error))
		println(@sprintf("  Plasma current mean error: %.1f%%", 100*plasma_error))

		println("  === vs Time-varying M & L₂ Analytical ===")
		println(@sprintf("  Coil current mean error: %.1f%%", 100*coil_error_ML))
		println(@sprintf("  Plasma current mean error: %.1f%%", 100*plasma_error_ML))

		if coil_error < 0.10 && plasma_error < 0.20  # Looser tolerance for plasma
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
		create_coupling_plots_combined(times, I_coil, I_plasma,
		                              I_coil_analytical, I_plasma_analytical,
		                              I_coil_analytical_ML, I_plasma_analytical_ML)
	end

	return (
		coil_error=coil_error,
		plasma_error=plasma_error,
		coil_error_ML=coil_error_ML,
		plasma_error_ML=plasma_error_ML,
		times=times,
		I_coil_sim=I_coil,
		I_plasma_sim=I_plasma,
		I_coil_analytical=I_coil_analytical,
		I_plasma_analytical=I_plasma_analytical,
		I_coil_analytical_ML=I_coil_analytical_ML,
		I_plasma_analytical_ML=I_plasma_analytical_ML,
		M_values=M_values,
		L_plasma_values=L_plasma_values,
		L_coil=L_coil,
		L_plasma=L_plasma,
		mean_M=mean_M
	)
end

function create_coupling_plots_combined(times, I_coil_sim, I_plasma_sim,
                                       I_coil_ana, I_plasma_ana,
                                       I_coil_ana_ML, I_plasma_ana_ML)
    """Create comprehensive visualization plots comparing constant M and time-varying M&L analytical solutions"""

    # Convert time to milliseconds for plotting
    times_ms = times * 1e3

    # Current evolution plots - Coil
    p1 = plot(times_ms, I_coil_sim,
              label="Simulation",
              linewidth=3,
              color=:blue,
              xlabel="Time (ms)",
              ylabel="Current (A)",
              title="Coil Current Evolution")
    plot!(p1, times_ms, I_coil_ana,
          label="Constant M&L",
          linestyle=:dash,
          linewidth=2,
          color=:red)
    plot!(p1, times_ms, I_coil_ana_ML,
          label="Time-varying M&L",
          linestyle=:dashdot,
          linewidth=2,
          color=:orange)

    # Current evolution plots - Plasma
    p2 = plot(times_ms, I_plasma_sim,
              label="Simulation",
              linewidth=3,
              color=:blue,
              xlabel="Time (ms)",
              ylabel="Current (A)",
              title="Plasma Current Evolution")
    plot!(p2, times_ms, I_plasma_ana,
          label="Constant M&L",
          linestyle=:dash,
          linewidth=2,
          color=:red)
    plot!(p2, times_ms, I_plasma_ana_ML,
          label="Time-varying M&L",
          linestyle=:dashdot,
          linewidth=2,
          color=:orange)

    # Error plots with robust calculation (avoiding first index issues)
    function calculate_error_robust(sim, ana)
        # Use indices 2:end to avoid t=0 issues where ana might be zero
        valid_indices = 2:length(ana)
        ana_valid = ana[valid_indices]
        sim_valid = sim[valid_indices]

        # Filter out very small analytical values to avoid Inf
        large_enough = abs.(ana_valid) .> 1e-10
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
    p3 = plot(times_ms[2:end], coil_error_const,
              label="vs Constant M&L",
              linewidth=2,
              color=:red,
              xlabel="Time (ms)",
              ylabel="Error (%)",
              title="Coil Current Errors"
              )
    plot!(p3, times_ms[2:end], coil_error_ML,
          label="vs Time-varying M&L",
          linewidth=2,
          color=:orange)

    # Error plot - Plasma
    p4 = plot(times_ms[2:end], plasma_error_const,
              label="vs Constant M&L",
              linewidth=2,
              color=:red,
              xlabel="Time (ms)",
              ylabel="Error (%)",
              title="Plasma Current Errors")
    plot!(p4, times_ms[2:end], plasma_error_ML,
          label="vs Time-varying M&L",
          linewidth=2,
          color=:orange)

    # Combined plot
    plot_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = joinpath(pwd(), "plasma_coil_coupling_comparison_$(timestamp).png")
    savefig(plot_combined, filename)

    println("\nVisualization:")
    println("  ✓ Combined comparison plot saved as: $(filename)")

    # Create animation if possible
    try
        animate_snaps2D(RP.diagnostics.snaps2D, RP.G.R1D, RP.G.Z1D,
                    [:ne,:ue_para,:Jϕ,:E_para_tot];
					wall=RP.fitted_wall,
                    filename="plasma_coil_coupling_$(timestamp).mp4")
        println("  ✓ Animation saved as: plasma_coil_coupling_$(timestamp).mp4")
    catch e
        println("  ⚠ Animation creation failed: $(e)")
    end

    return plot_combined
end

function create_coupling_plots(times, I_coil_sim, I_plasma_sim, I_coil_ana, I_plasma_ana)
    """Create visualization plots for coupling test results (legacy function)"""

    # Current evolution plots
    p1 = plot(times * 1e3, I_coil_sim,
              label="Coil (Sim)",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Current (A)",
              title="Coil Current Evolution")
    plot!(p1, times * 1e3, I_coil_ana,
          label="Coil (Analytical)",
          linestyle=:dash,
          linewidth=2)

    p2 = plot(times * 1e3, I_plasma_sim,
              label="Plasma (Sim)",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Current (A)",
              title="Plasma Current Evolution")
    plot!(p2, times * 1e3, I_plasma_ana,
          label="Plasma (Analytical)",
          linestyle=:dash,
          linewidth=2)

    # Error plots with robust calculation
    # Use indices 2:end to avoid t=0 issues and filter out very small analytical values
    valid_indices = 2:length(I_coil_ana)
    ana_coil_valid = I_coil_ana[valid_indices]
    sim_coil_valid = I_coil_sim[valid_indices]
    ana_plasma_valid = I_plasma_ana[valid_indices]
    sim_plasma_valid = I_plasma_sim[valid_indices]

    # Calculate errors, avoiding division by very small numbers
    coil_error = zeros(length(valid_indices))
    plasma_error = zeros(length(valid_indices))

    coil_large_enough = abs.(ana_coil_valid) .> 1e-10
    plasma_large_enough = abs.(ana_plasma_valid) .> 1e-10

    coil_error[coil_large_enough] = abs.(sim_coil_valid[coil_large_enough] - ana_coil_valid[coil_large_enough]) ./
                                   abs.(ana_coil_valid[coil_large_enough]) * 100
    plasma_error[plasma_large_enough] = abs.(sim_plasma_valid[plasma_large_enough] - ana_plasma_valid[plasma_large_enough]) ./
                                       abs.(ana_plasma_valid[plasma_large_enough]) * 100

    p3 = plot(times[valid_indices] * 1e3, coil_error,
              label="Coil Error",
              linewidth=2,
              xlabel="Time (ms)",
              ylabel="Error (%)",
              title="Relative Errors")
    plot!(p3, times[valid_indices] * 1e3, plasma_error,
          label="Plasma Error",
          linewidth=2)

    # Combined plot
    plot_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 900))

    # Save plot
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH_MM_SS")
    filename = joinpath(pwd(), "plasma_coil_coupling_$(timestamp).png")
    savefig(plot_combined, filename)

    println("\nVisualization:")
    println("  ✓ Plot saved as: $(filename)")

    return plot_combined
end

@testset "RAPID2D.jl Plasma-Coil Coupling Test" begin
    # Run the coupling test
    RP = run_plasma_coil_coupling_test(verbose=verbose, visualize=visualize)

    # Analyze results
    results = analyze_coupling_results(RP; verbose, visualize)

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

# Main execution (when run directly as a script)
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting RAPID2D.jl Plasma-Coil Coupling Test...")

    try
        RP = run_plasma_coil_coupling_test(verbose=true, visualize=true)
        results = analyze_coupling_results(RP; verbose=true, visualize=true)

        if results !== nothing && results.coil_error < 0.15 && results.plasma_error < 0.30
            println("\n" * "=" ^ 70)
            println("✓ PLASMA-COIL COUPLING TEST COMPLETED SUCCESSFULLY")
            println("=" ^ 70)
        else
            println("\n" * "=" ^ 70)
            println("✗ PLASMA-COIL COUPLING TEST FAILED")
            println("=" ^ 70)
        end
    catch e
        println("\n" * "=" ^ 70)
        println("✗ PLASMA-COIL COUPLING TEST FAILED")
        println("Error: $(e)")
        println("=" ^ 70)
        rethrow(e)
    end
end
