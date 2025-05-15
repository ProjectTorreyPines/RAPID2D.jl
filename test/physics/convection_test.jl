using Test
using LinearAlgebra
using SparseArrays
using RAPID2D

@testset "Convection Term Calculation - Explicit vs Implicit" begin
    # Define test parameters
    NR, NZ = 15, 30  # Small grid for testing
    FT = Float64     # Floating point type

    # Create a proper RAPID object for testing using create_rapid_object
    config = RAPID2D.SimulationConfig{FT}()
    config.prefilled_gas_pressure = 4e-3
    config.R0B0 = 1.5*1.8
    config.NR = NR
    config.NZ = NZ

    # Create the RAPID object
    RP = create_rapid_object(; config=config)

    # Create mock density field with a smooth Gaussian-like profile
    test_density = zeros(FT, NR, NZ)
    for j in 1:NZ, i in 1:NR
        r_norm = (RP.G.R1D[i] - 1.5)^2 / 0.3^2
        z_norm = RP.G.Z1D[j]^2 / 0.2^2
        test_density[i,j] = exp(-(r_norm + z_norm))
    end

    # Create mock velocity fields with spatial variation to test all code paths
    # Including regions with positive, negative, and near-zero velocities
    uR = zeros(FT, NR, NZ)
    uZ = zeros(FT, NR, NZ)

    for j in 1:NZ, i in 1:NR
        # R-direction velocity: varies from negative to positive
        uR[i,j] = 0.2 * (i - NR/2) / (NR/2)

        # Z-direction velocity: radial pattern from center
        uZ[i,j] = 0.1 * (j - NZ/2) / (NZ/2)

        # Create some regions with near-zero velocity to test that case
        if abs(uR[i,j]) < 0.05
            uR[i,j] = 0.0
        end
        if abs(uZ[i,j]) < 0.05
            uZ[i,j] = 0.0
        end
    end

    # Set velocities in the RAPID object
    RP.plasma.ueR .= uR
    RP.plasma.ueZ .= uZ

    # Test both upwind scheme and central differencing
    for flag_upwind in [true, false]
        @testset "Upwind = $flag_upwind" begin
            # Calculate explicit convection term
            RAPID2D.calculate_convection_term!(RP, test_density, uR, uZ, flag_upwind)
            explicit_result = copy(RP.operators.neRHS_convec)

            # Calculate implicit convection term
            An_convec = RAPID2D.construct_convection_operator!(RP, uR, uZ, flag_upwind)
            implicit_result = reshape(An_convec * test_density[:], NR, NZ)

            # Compare the results
            @test isapprox(explicit_result, implicit_result, rtol=1e-10)

            # Additional test: Verify that matrix multiplication operations don't error
            @test_nowarn An_convec * (An_convec * test_density[:])
        end
    end

    # Additional test: Create a more complex velocity field with vortices
    @testset "Complex Velocity Field" begin
        # Create a vortex-like flow pattern
        for j in 1:NZ, i in 1:NR
            # Centered coordinates
            x = i - NR/2
            y = j - NZ/2
            r = sqrt(x^2 + y^2)

            if r > 0
                # Circular flow around center
                uR[i,j] = -0.1 * y / r
                uZ[i,j] = 0.1 * x / r
            else
                uR[i,j] = 0.0
                uZ[i,j] = 0.0
            end
        end

        RP.plasma.ueR .= uR
        RP.plasma.ueZ .= uZ

        # Test with upwind scheme (more stable for complex flows)
        RAPID2D.calculate_convection_term!(RP, test_density, uR, uZ, true)
        explicit_result = copy(RP.operators.neRHS_convec)

        An_convec = RAPID2D.construct_convection_operator!(RP, uR, uZ, true)
        implicit_result = reshape(An_convec * test_density[:], NR, NZ)

        @test isapprox(explicit_result, implicit_result, rtol=1e-10)
    end
end