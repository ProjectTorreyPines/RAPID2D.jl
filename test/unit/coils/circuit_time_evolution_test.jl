using RAPID2D
using Test
using RAPID2D.LinearAlgebra

function simulate_circuit_evolution(csys::CoilSystem{FT}, t_span::Tuple{FT, FT},
                                   dt::FT; save_history::Bool=true) where FT<:AbstractFloat
    t_start, t_end = t_span
    @assert t_end > t_start "End time must be greater than start time"
    @assert dt > 0 "Time step must be positive"

    # Set time step in system
	csys.time_s = t_start
    csys.Δt = dt

    # Ensure matrices are computed
    if size(csys.A_LR_circuit, 1) != csys.n_total
        update_coil_system_matrices!(csys)
    end

    N_steps = ceil(Int, (t_end - t_start) / dt)+1

    if save_history
        time_history = zeros(FT, N_steps)
        current_history = zeros(FT, N_steps, csys.n_total)
    end

    for i in 1:N_steps
        if save_history
            time_history[i] = csys.time_s
            current_history[i, :] = get_all_currents(csys)
        end

        # Advance one time step
        advance_LR_circuit_step!(csys)
    end

    if save_history
        return (time_history=time_history, current_history=current_history)
    else
        return nothing
    end
end

# Calculate analytical solution for comparison
function analytical_coupled_LR_solution(t, L1, L2, M, R1, R2, V1, V2)
	# System: L * dI/dt + R * I = V
	# Transform to: dI/dt = -L⁻¹R * I + L⁻¹V
	# Solution: I(t) = I_steady + c1*v1*exp(λ1*t) + c2*v2*exp(λ2*t)

	L_matrix = [L1 M; M L2]
	R_matrix = [R1 0.0; 0.0 R2]
	V_vector = [V1; V2]

	# System matrix A = -L⁻¹R
	A_sys = -L_matrix \ R_matrix

	# Eigenvalues and eigenvectors of system matrix
	eigenvals, eigenvecs = eigen(A_sys)
	λ1, λ2 = eigenvals[1], eigenvals[2]
	v1, v2 = eigenvecs[:, 1], eigenvecs[:, 2]

	# Steady state solution: I_steady = R⁻¹ * V
	I_steady = R_matrix \ V_vector

	# Initial conditions: I(0) = 0, so I_steady + c1*v1 + c2*v2 = 0
	# This gives us: c1*v1 + c2*v2 = -I_steady
	# We need to solve the 2x2 system: [v1 v2] * [c1; c2] = -I_steady
	coeff_matrix = [v1 v2]
	coeffs = coeff_matrix \ (-I_steady)
	c1, c2 = coeffs[1], coeffs[2]

	# Solution at time t
	I_t = I_steady + c1 * v1 * exp(λ1 * t) + c2 * v2 * exp(λ2 * t)

	return I_t[1], I_t[2]
end

FT = Float64

@testset "Circuit Time Evolution" begin
	@testset "Single RL Circuit Step Response" begin
		# Test case: single RL circuit with step voltage input
		# Theoretical solution: I(t) = (V/R) * (1 - exp(-t*R/L))

		R = 1.0    # Resistance [Ω]
		L = 1e-3   # Inductance [H]
		V = 10.0   # Step voltage [V]

		coils = [
			Coil{FT}(
				location=(r=1.0, z=0.0),
				area=π*0.1^2,
				resistance=R,
				self_inductance=L,
				is_powered=true,
				name="rl_coil",
				current=0.0,      # Initial current
				voltage_ext=V     # Constant voltage
			)
		]

		csys = CoilSystem{FT}(coils)

		# Set up time parameters
		dt = 1e-6
		t_start = 0.0
		t_end = 5e-3  # 5 time constants (L/R = 1ms)

		# analytical solution
		time_s = t_start:dt:t_end
		I_analytic = @. (V / R) * (1.0 - exp(-time_s /(L/R)))

		for θimp in [0.0, 0.5, 1.0]
			csys.Δt = dt
			csys.θimp = θimp
			set_all_currents!(csys, zeros(FT, csys.n_total))  # Reset currents
			update_coil_system_matrices!(csys)

			result = simulate_circuit_evolution(csys, (0.0, t_end), dt)

			I_numerical = result.current_history[:, 1]

			if θimp in [0.0, 1.0]
				@test isapprox(I_numerical, I_analytic,rtol=1e-3)
				# For θimp = 0.0 or 1.0, we expect a less accurate solution
				@test !isapprox(I_numerical, I_analytic,rtol=1e-5)
			elseif θimp == 0.5
				# For θimp = 0.5, we expect a more accurate solution
				@test isapprox(I_numerical, I_analytic,rtol=1e-5)
			end
		end
	end

	@testset "Two Coupled Coils" begin
		# Test mutual inductance effects between two coils

		L1 = 1e-3    # Self inductance coil 1 [H]
		L2 = 1.5e-3  # Self inductance coil 2 [H]
		M = 0.5e-3   # Mutual inductance [H]
		R1 = R2 = 1.0  # Resistances [Ω]

		# Create coils positioned to give desired mutual inductance
		coils = [
			Coil{FT}(
				location=(r=1.0, z=0.0),
				area=π*0.1^2,
				resistance=R1,
				self_inductance=L1,
				is_powered=true,
				name="coil1",
				current=0.0,
				voltage_ext=10.0  # Only coil 1 has voltage
			),
			Coil{FT}(
				location=(r=1.5, z=0.0),
				area=π*0.1^2,
				resistance=R2,
				self_inductance=L2,
				is_powered=true,
				name="coil2",
				current=0.0,
				voltage_ext=0.0   # Coil 2 has no external voltage
			)
		]

		csys = CoilSystem{FT}(coils)
		csys.θimp = 1.0

		# Manually set mutual inductance (since actual Green's function calculation
		# would depend on detailed geometry)
		csys.mutual_inductance = [L1 M; M L2]

		dt = 1e-6
		t_end = 10e-3
		csys.Δt = dt
		calculate_circuit_matrices!(csys)

		result = simulate_circuit_evolution(csys, (0.0, t_end), dt)

		# Test that mutual coupling works:
		# 1. Current in coil 2 should be non-zero (induced by coil 1)
		# 2. Current in coil 2 should be opposite to coil 1 (Lenz's law)
		final_I1 = result.current_history[end, 1]
		final_I2 = result.current_history[end, 2]

		@test final_I1 > 0  # Coil 1 should have positive current
		@test final_I2 < 0  # Coil 2 should have negative current (opposing)
		@test abs(final_I2) < abs(final_I1)  # Coil 2 current should be smaller
	end

	@testset "Energy and Power Calculations" begin
		# Test energy and power calculation functions

		R = 1.0
		L = 1e-3

		coils = [
			Coil{FT}(
				location=(r=1.0, z=0.0),
				area=π*0.1^2,
				resistance=R,
				self_inductance=L,
				is_powered=true,
				name="test_coil",
				current=2.0,
				voltage_ext=0.0
			)
		]

		csys = CoilSystem{FT}(coils)
		update_coil_system_matrices!(csys)

		# Test energy calculation: E = 0.5 * L * I^2
		energy = calculate_circuit_magnetic_energy(csys)
		expected_energy = 0.5 * L * 2.0^2
		@test energy ≈ expected_energy

		# Test power calculation: P = R * I^2
		power = calculate_power_dissipation(csys)
		expected_power = R * 2.0^2
		@test power ≈ expected_power
	end

	@testset "Two Coupled Coils - Analytical Comparison" begin
		# Test coupled coils system against analytical solution
		# Circuit equations: L₁(dI₁/dt) + M(dI₂/dt) + R₁I₁ = V₁
		#                   L₂(dI₂/dt) + M(dI₁/dt) + R₂I₂ = V₂

		L1 = 1e-3    # Self inductance coil 1 [H]
		L2 = 1.5e-3  # Self inductance coil 2 [H]
		M = 0.5e-3   # Mutual inductance [H]
		R1 = R2 = 1.0  # Resistances [Ω]
		V1, V2 = 10.0, 0.0  # Voltages [V]

		# Verify coupling coefficient is physical (k < 1)
		k = M / sqrt(L1 * L2)
		@test k < 1.0  # Coupling coefficient must be less than 1

		# Create coils for coupled system
		coils = [
			Coil{FT}(
				location=(r=1.0, z=0.0),
				area=π*0.1^2,
				resistance=R1,
				self_inductance=L1,
				is_powered=true,
				name="coil1",
				current=0.0,
				voltage_ext=V1
			),
			Coil{FT}(
				location=(r=1.5, z=0.0),
				area=π*0.1^2,
				resistance=R2,
				self_inductance=L2,
				is_powered=true,
				name="coil2",
				current=0.0,
				voltage_ext=V2
			)
		]

		csys = CoilSystem{FT}(coils)
		csys.θimp = 0.5  # Trapezoidal rule for better accuracy

		# Manually set mutual inductance for controlled test
		csys.mutual_inductance = [L1 M; M L2]

		dt = 1e-6
		t_end = 1e-3  # Longer simulation to reach steady state
		csys.Δt = dt
		calculate_circuit_matrices!(csys)

		result = simulate_circuit_evolution(csys, (0.0, t_end), dt)



		# Calculate analytical solution at all time points
		time_points = result.time_history
		I1_analytical = zeros(length(time_points))
		I2_analytical = zeros(length(time_points))

		for (i, t) in enumerate(time_points)
			I1_analytical[i], I2_analytical[i] = analytical_coupled_LR_solution(
				t, L1, L2, M, R1, R2, V1, V2)
		end

		# Compare numerical and analytical solutions
		I1_numerical = result.current_history[:, 1]
		I2_numerical = result.current_history[:, 2]

		@test isapprox(I1_numerical, I1_analytical, rtol=1e-4)
		@test isapprox(I2_numerical, I2_analytical, rtol=1e-4)
	end

	@testset "Energy Conservation in coupled LR System" begin
		# Verify energy conservation in coupled coil system
		L1, L2, M = 1e-3, 1.5e-3, 0.5e-3
		R1, R2 = 1.0, 1.0
		V1, V2 = 5.0, 0.0

		coils = [
			Coil{FT}(
				location=(r=1.0, z=0.0),
				area=π*0.1^2,
				resistance=R1,
				self_inductance=L1,
				is_powered=true,
				name="coil1",
				current=0.0,
				voltage_ext=V1
			),
			Coil{FT}(
				location=(r=1.5, z=0.0),
				area=π*0.1^2,
				resistance=R2,
				self_inductance=L2,
				is_powered=true,
				name="coil2",
				current=0.0,
				voltage_ext=V2
			)
		]

		csys = CoilSystem{FT}(coils)
		csys.θimp = 0.5
		csys.mutual_inductance = [L1 M; M L2]

		dt = 10e-6
		t_end = 1e-3
		csys.Δt = dt
		calculate_circuit_matrices!(csys)

		result = simulate_circuit_evolution(csys, (0.0, t_end), dt)

		# Calculate energy and power at each time step
		N_steps = length(result.time_history)
		magnetic_energy = zeros(N_steps)
		power_dissipated = zeros(N_steps)
		power_input = zeros(N_steps)

		for i in 1:N_steps
			I1, I2 = result.current_history[i, 1], result.current_history[i, 2]

			# Magnetic energy: E = 0.5 * [I1 I2] * [L1 M; M L2] * [I1; I2]
			magnetic_energy[i] = 0.5 * (L1*I1^2 + 2*M*I1*I2 + L2*I2^2)
			# Power dissipated: P_loss = I1²R1 + I2²R2
			power_dissipated[i] = I1^2*R1 + I2^2*R2
			# Power input: P_in = V1*I1 + V2*I2
			power_input[i] = V1*I1 + V2*I2
		end

		# Energy conservation: ∫(P_input-P_dissipated) dt = magnetic energy changes
		integrated_energy_change = sum(power_input - power_dissipated) * dt
		@test isapprox(integrated_energy_change, magnetic_energy[end] - magnetic_energy[1], rtol=1e-2)

		# Additional check: total energy should increase initially then stabilize
		@test magnetic_energy[end] > magnetic_energy[1]  # Energy increases from zero
		@test magnetic_energy[end] > magnetic_energy[end÷2]  # Energy continues to increase
	end

	@testset "Zero Coil Systems" begin
		# Test behavior with empty coil system
		csys = CoilSystem{FT}()

		@test csys.n_total == 0

		# Test that functions handle empty systems gracefully
		calculate_mutual_inductance_matrix!(csys)
		calculate_circuit_matrices!(csys)

		@test calculate_circuit_magnetic_energy(csys) ≈ 0.0
		@test calculate_power_dissipation(csys) ≈ 0.0
	end
end

@testset "RAPID's coils evolution without plasma" begin

    config = SimulationConfig{FT}(
		NR = 30, NZ =50,
		prefilled_gas_pressure = 0.0,
		R0B0 = 2.5,
		device_Name = "manual",
		dt = 20e-6,
		snap0D_Δt_s = 20e-6,
		snap2D_Δt_s = 500e-6,
		t_end_s = 5e-3,
	)

	flags = SimulationFlags{FT}(
		Gas_evolve = false,
		Ampere = true,
		Ampere_Itor_threshold = 0.0
	)
	@testset "Two Coupled Coils - Analytical Comparison" begin
		# Test coupled coils system against analytical solution
		# Circuit equations: L₁(dI₁/dt) + M(dI₂/dt) + R₁I₁ = V₁
		#                   L₂(dI₂/dt) + M(dI₁/dt) + R₂I₂ = V₂

		RP = RAPID(config)
		RP.flags = flags

		coil_area = π * (0.01)^2
		resistivity = 1e8*1.68e-8

		μ0 = 4π * 1e-7

		coil_location1 = (r=1.0, z=0.0)
		coil_location2 = (r=1.5, z=0.0)

		coil_resistance1 = calculate_coil_resistance(coil_location1.r, coil_area, resistivity)
		coil_resistance2 = calculate_coil_resistance(coil_location2.r, coil_area, resistivity)

		coil_L_self_1 = calculate_self_inductance(coil_area, coil_location1.r, μ0)
		coil_L_self_2 = calculate_self_inductance(coil_area, coil_location2.r, μ0)

        # Create coils for coupled system
        coils = [
            Coil{FT}(
                location=coil_location1,
                area=coil_area,
                resistance=coil_resistance1,
                self_inductance=coil_L_self_1,
                is_powered=true,
                name="coil1",
                current=0.0,
                voltage_ext=10.0
            ),
            Coil{FT}(
                location=coil_location2,
                area=coil_area,
                resistance=coil_resistance2,
                self_inductance=coil_L_self_2,
                is_powered=true,
                name="coil2",
                current=0.0,
                voltage_ext=0.0
            )
        ]

		add_coil!(RP.coil_system, coils[1])
		add_coil!(RP.coil_system, coils[2])

		initialize!(RP)

		RP.plasma.ne .= 0.0
		RP.plasma.ni .= 0.0
		update_transport_quantities!(RP)


		run_simulation!(RP)

		M = RP.coil_system.mutual_inductance[1,2]
		L1 = coils[1].self_inductance
		L2 = coils[2].self_inductance
		# Verify coupling coefficient is physical (k < 1)
		@test M / sqrt(L1 * L2) < 1.0  # Coupling coefficient must be less than 1

		# Calculate analytical solution at all time points
		time_points = RP.diagnostics.snaps0D.time_s
		I1_analytical = zeros(length(time_points))
		I2_analytical = zeros(length(time_points))

		for (i, t) in enumerate(time_points)
			I1_analytical[i], I2_analytical[i] = analytical_coupled_LR_solution(
				t, coils[1].self_inductance, coils[2].self_inductance,
				RP.coil_system.mutual_inductance[1,2],
				coils[1].resistance, coils[2].resistance,
				coils[1].voltage_ext, coils[2].voltage_ext)
		end

		# Compare numerical and analytical solutions
		I1_numerical = RP.diagnostics.snaps0D.coils_I[1,:]
		I2_numerical = RP.diagnostics.snaps0D.coils_I[2,:]

		@test isapprox(I1_numerical, I1_analytical, rtol=1e-2)
		@test isapprox(I2_numerical, I2_analytical, rtol=1e-2)


		# Test energy conservation
		snaps0D = RP.diagnostics.snaps0D
		W_input = sum(snaps0D.tot_P_input_coils)*RP.config.snap0D_Δt_s
		W_ohm_coil = sum(snaps0D.tot_P_ohm_coils)*RP.config.snap0D_Δt_s
		W_ohm_plasma = sum(snaps0D.tot_P_ohm_plasma)*RP.config.snap0D_Δt_s
		ΔW_mag = snaps0D.tot_W_mag[end] - snaps0D.tot_W_mag[1]

		ΔW = W_input - (W_ohm_coil + W_ohm_plasma + ΔW_mag)  # should be close to zero
		@test abs(ΔW) / abs(W_input) < 0.01  # within 1%
	end
end