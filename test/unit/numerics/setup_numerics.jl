# Shared @testsnippet fixtures for test/unit/numerics/.

@testsnippet NumericsFixtures begin
    # The dominant config shape in this directory: a rectangular vacuum vessel
    # (wall_R/wall_Z) inside a 0.8-2.2 m x -1.2-1.2 m domain. NR/NZ/dt/t_end_s/
    # Dpara0/Dperp0 are parameterized because callers vary them; the geometry, R0B0
    # and prefilled_gas_pressure are what make this "the walled box" and are fixed.
    function walled_box_config(
            ::Type{FT} = Float64;
            NR::Int = 100, NZ::Int = 200,
            dt = 1.0e-6, t_end_s = 10.0e-6,
            Dpara0 = 0.0, Dperp0 = 0.0
        ) where {FT <: AbstractFloat}
        return SimulationConfig{FT}(
            NR = NR, NZ = NZ,
            R_min = 0.8, R_max = 2.2,
            Z_min = -1.2, Z_max = 1.2,
            dt = dt, t_end_s = t_end_s,
            R0B0 = 1.0,
            prefilled_gas_pressure = 5.0e-3,
            wall_R = FT[1.0, 2.0, 2.0, 1.0],
            wall_Z = FT[-1.0, -1.0, 1.0, 1.0],
            Dpara0 = Dpara0, Dperp0 = Dperp0,
        )
    end

    # The wall-free, prefilled-gas config used by the operator-construction tests.
    # NR=50/NZ=100 are the SimulationConfig struct defaults; the ∇𝐃∇ and ∇⋅(𝐮 f)
    # testitems pass NR=15, NZ=30.
    function gas_filled_config(
            ::Type{FT} = Float64;
            NR::Int = 50, NZ::Int = 100
        ) where {FT <: AbstractFloat}
        return SimulationConfig{FT}(
            prefilled_gas_pressure = 4.0e-3,
            R0B0 = 1.5 * 1.8,
            NR = NR,
            NZ = NZ,
        )
    end
end

@testsnippet DiscretizedOperatorFixtures begin
    # A fresh (identity, central-difference-in-R) DiscretizedOperator pair on a 3x3
    # grid — 9 unknowns, so both operators are 9x9. Returned fresh on every call, so
    # callers may mutate the results without affecting each other.
    function create_test_operators()
        NR, NZ = 3, 3
        dims_rz = (NR, NZ)

        # Identity-like operator
        I_id = Int[]
        J_id = Int[]
        V_id = Float64[]

        # First derivative in R direction (central difference)
        I_dr = Int[]
        J_dr = Int[]
        V_dr = Float64[]

        # Fill the identity operator
        for i in 1:(NR * NZ)
            push!(I_id, i)
            push!(J_id, i)
            push!(V_id, 1.0)
        end

        # Fill a central difference operator for dR
        # (only interior points, boundary points are zero)
        k = 1
        for j in 1:NZ
            for i in 1:NR
                if i > 1 && i < NR
                    # East neighbor
                    push!(I_dr, k)
                    push!(J_dr, k + 1)
                    push!(V_dr, 0.5)

                    # West neighbor
                    push!(I_dr, k)
                    push!(J_dr, k - 1)
                    push!(V_dr, -0.5)
                end
                k += 1
            end
        end

        # Create the operators
        id_op = DiscretizedOperator(dims_rz, I_id, J_id, V_id)
        dr_op = DiscretizedOperator(dims_rz, I_dr, J_dr, V_dr)

        return id_op, dr_op
    end
end
