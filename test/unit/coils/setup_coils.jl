# Shared setup for the test/unit/coils/ suite.
#
# The neutral `setup_*.jl` name is load-bearing. Do NOT rename to `*_testsetup.jl`:
# that suffix matches ReTestItems' `is_testsetup_file`, so the parallel path
# (test/runtests_parallel.jl) would hand this file to ReTestItems, which understands
# `@testsetup module` but NOT `@testsnippet`, and dies.
#
# Everything here is a FACTORY FUNCTION, never a `const` instance: call sites mutate
# `.current` / `.voltage_ext` on the coils they receive, so a shared mutable constant
# would leak state across testitems.

@testsnippet CoilFactories begin
    # Canonical powered PF-style coil: area 0.01 m², R 1 mΩ, L 1 μH.
    # Only the fields that actually vary between call sites are exposed as kwargs.
    pf_coil(
        name; r = 2.5, z = 0.5, is_controllable = true,
        max_voltage = 1000.0, max_current = 50000.0,
        current = 0.0, voltage_ext = 0.0
    ) =
        Coil(
        (r = r, z = z), 0.01, 0.001, 1.0e-6, true, is_controllable, name,
        max_voltage, max_current, current, voltage_ext
    )

    # Canonical passive wall segment: area 0.005 m², R 10 mΩ, L 0.1 μH.
    # Never powered and never controllable, hence no current/voltage limits.
    wall_coil(name; r = 2.01, z = 0.0, current = 0.0, voltage_ext = 0.0) =
        Coil(
        (r = r, z = z), 0.005, 0.01, 1.0e-7, false, false, name,
        nothing, nothing, current, voltage_ext
    )
end

@testsnippet CoilGridHelpers begin
    # Geometrically trivial coil (area = resistance = self-inductance = 1) so the
    # current-distribution tests observe the raw bilinear weighting rather than a
    # scaled version of it.
    unit_coil(r, z, name; is_controllable = true) =
        Coil{Float64}(
        location = (r = r, z = z), area = 1.0, resistance = 1.0,
        self_inductance = 1.0, is_powered = true,
        is_controllable = is_controllable, name = name
    )

    function make_grid(NR, NZ, r_span, z_span)
        grid = GridGeometry{Float64}(NR, NZ)
        initialize_grid_geometry!(grid, r_span, z_span)
        return grid
    end

    function place_coils(grid, coils::AbstractVector)
        coil_system = CoilSystem{Float64}(coils)
        determine_coils_inside_grid!(coil_system, grid)
        return coil_system
    end
    place_coils(grid, coil::Coil) = place_coils(grid, [coil])
end

@testsnippet CircuitHelpers begin
    using LinearAlgebra  # analytical_coupled_LR_solution needs `eigen`

    function simulate_circuit_evolution(
            csys::CoilSystem{FT}, t_span::Tuple{FT, FT},
            dt::FT; save_history::Bool = true
        ) where {FT <: AbstractFloat}
        t_start, t_end = t_span
        @assert t_end > t_start "End time must be greater than start time"
        @assert dt > 0 "Time step must be positive"

        # Set time step in system
        csys.time_s = t_start
        csys.Δt = dt

        # Ensure matrices are computed.
        # NOTE: this guard is why callers that hand-assign csys.mutual_inductance must
        # call calculate_circuit_matrices!(csys) themselves BEFORE getting here —
        # otherwise update_coil_system_matrices! would recompute and destroy the override.
        if size(csys.A_LR_circuit, 1) != csys.n_total
            update_coil_system_matrices!(csys)
        end

        N_steps = ceil(Int, (t_end - t_start) / dt) + 1

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
            return (time_history = time_history, current_history = current_history)
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
end
