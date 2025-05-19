using Test
using LinearAlgebra
using SparseArrays
using RAPID2D

@testset "Diffusion Term Calculation - Explicit vs Implicit" begin
    # Define test parameters
    NR, NZ = 15, 30  # Small grid for testing
    FT = Float64    # Floating point type

    # Create a proper RAPID object for testing using create_rapid_object
    config = RAPID2D.SimulationConfig{FT}()
	config.prefilled_gas_pressure = 4e-3
	config.R0B0 = 1.5*1.8;
	config.NR = NR
	config.NZ = NZ

    # Create the RAPID object
    RP = create_rapid_object(; config=config)

    # Create mock diffusion coefficients with spatial variation
    for j in 1:NZ, i in 1:NR
        RP.transport.DRR[i,j] = 0.1 + 0.01 * (i + j)  # Vary with position
        RP.transport.DZZ[i,j] = 0.2 + 0.01 * (i + j)  # Vary with position
        RP.transport.DRZ[i,j] = 0.01 * (i - j)        # Vary with position, can be negative
    end

    # Create mock density field with a smooth Gaussian-like profile
    test_density = zeros(FT, NR, NZ)
    for j in 1:NZ, i in 1:NR
        r_norm = (RP.G.R1D[i] - 1.5)^2 / 0.3^2
        z_norm = RP.G.Z1D[j]^2 / 0.2^2
        test_density[i,j] = exp(-(r_norm + z_norm))
    end

	# Explicit method
    RAPID2D.calculate_ne_diffusion_explicit_RHS!(RP, test_density)
    explicit_result = copy(RP.operators.neRHS_diffu)

    # Implicit method
    An_diffu = RAPID2D.construct_∇𝐃∇_operator(RP)
    implicit_result = reshape(An_diffu * test_density[:], NR, NZ)

    # Calculate implicit diffusion using RAPID2D's internal way that can update the operator more efficiently
    # This is useful for large simulations where we want to avoid re-creating the operator
    RAPID2D.initialize_diffusion_operator!(RP)
    implicit_result2 = reshape(RP.operators.An_diffu * test_density[:], NR, NZ)

    # compare if two methods give the same operatoryy
    @test An_diffu == RP.operators.An_diffu

	# Comparison
	@test isapprox(explicit_result, implicit_result, rtol=1e-10)
	@test isapprox(explicit_result, implicit_result2, rtol=1e-10)


    # Additional test: Verify that matrix multiplication operations don't error
    # This is to ensure our implementation works with typical linear algebra operations
    @test_nowarn An_diffu * (An_diffu * test_density[:])
end