using RAPID2D
using LinearAlgebra
using SparseArrays
using Test

@testset "Diffusion operator [∇𝐃∇] - Explicit vs Implicit" begin
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
    A_∇𝐃∇ = RAPID2D.construct_∇𝐃∇_operator(RP)
    implicit_result = reshape(A_∇𝐃∇ * test_density[:], NR, NZ)

    # Calculate implicit diffusion using RAPID2D's internal way that can update the operator more efficiently
    # This is useful for large simulations where we want to avoid re-creating the operator
    RAPID2D.initialize_∇𝐃∇_operator!(RP)
    implicit_result2 = reshape(RP.operators.A_∇𝐃∇ * test_density[:], NR, NZ)

    # compare if two methods give the same operatoryy
    @test A_∇𝐃∇ == RP.operators.A_∇𝐃∇

	# Comparison
	@test isapprox(explicit_result, implicit_result, rtol=1e-10)
	@test isapprox(explicit_result, implicit_result2, rtol=1e-10)


    # Additional test: Verify that matrix multiplication operations don't error
    # This is to ensure our implementation works with typical linear algebra operations
    @test_nowarn A_∇𝐃∇ * (A_∇𝐃∇ * test_density[:])
end

@testset "Convection operator [-∇⋅(nu)]  - Explicit vs Implicit" begin
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
            RAPID2D.calculate_ne_convection_explicit_RHS!(RP, test_density, uR, uZ; flag_upwind)
            explicit_result = copy(RP.operators.neRHS_convec)

            # Calculate implicit convection term
            An_convec = RAPID2D.construct_Ane_convection_operator(RP, uR, uZ; flag_upwind)
            implicit_result = reshape(An_convec * test_density[:], NR, NZ)

            # Calculate implicit convection using RAPID2D's internal way that can update the operator more efficiently
            # This is useful for large simulations where we want to avoid re-creating the operator
            RAPID2D.initialize_Ane_convection_operator!(RP; flag_upwind)
            implicit_result2 = reshape(RP.operators.An_convec * test_density[:], NR, NZ)

            # compare if two methods give the same operatoryy
            @test An_convec == RP.operators.An_convec

            # Compare the results
            @test isapprox(explicit_result, implicit_result, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result2, rtol=1e-10)

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
        RAPID2D.calculate_ne_convection_explicit_RHS!(RP, test_density, uR, uZ; flag_upwind=true)
        explicit_result = copy(RP.operators.neRHS_convec)

        An_convec = RAPID2D.construct_Ane_convection_operator(RP, uR, uZ; flag_upwind=true)
        implicit_result = reshape(An_convec * test_density[:], NR, NZ)

        RAPID2D.initialize_Ane_convection_operator!(RP; flag_upwind=true)
        implicit_result2 = reshape(RP.operators.An_convec * test_density[:], NR, NZ)

        @test isapprox(explicit_result, implicit_result, rtol=1e-10)
        @test isapprox(explicit_result, implicit_result2, rtol=1e-10)
    end
end


@testset "[𝐮⋅∇] operator - Explicit vs Implicit" begin

    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=100, NZ=200,
        R_min=0.8, R_max=2.2,
        Z_min=-1.2, Z_max=1.2,
        dt=1e-6, t_end_s=10e-6,
        R0B0=1.0,
        prefilled_gas_pressure=5e-3,
        wall_R=[1.0, 2.0, 2.0, 1.0],
        wall_Z=[-1.0, -1.0, 1.0, 1.0]
    )

    # Create RAPID object
    RP = RAPID{FT}(config)
    initialize!(RP)

    NR, NZ = RP.G.NR, RP.G.NZ

    # Set up a specific siutation for testing
    RP.fields.BR_ext .= 20e-4
    RP.fields.BZ_ext .= 10e-4
    RAPID2D.combine_external_and_self_fields!(RP)

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.2
    sigma_Z = 0.2
    peak_ue_para = 1.0e6  # Peak velocity
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
        RP.plasma.ue_para[i, j] = peak_ue_para * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    RAPID2D.update_transport_quantities!(RP)

    # Test both upwind scheme and central differencing
    for flag_upwind in [true, false]
        @testset "Upwind = $flag_upwind" begin
            # Calculate explicit convection term
			∇ud_R, ∇ud_Z = calculate_grad_of_scalar_F(RP, RP.plasma.ue_para; upwind=flag_upwind)
			explicit_result = @. (RP.plasma.ueR*∇ud_R + RP.plasma.ueZ*∇ud_Z)

            # Calculate implicit convection term
            A_𝐮∇ = RAPID2D.construct_𝐮∇_operator(RP; flag_upwind)
            implicit_result = reshape(A_𝐮∇ * RP.plasma.ue_para[:], NR, NZ)

            # Calculate implicit convection using RAPID2D's internal way that can update the operator more efficiently
            # This is useful for large simulations where we want to avoid re-creating the operator
            RAPID2D.initialize_𝐮∇_operator!(RP; flag_upwind)
            implicit_result2 = reshape(RP.operators.A_𝐮∇ * RP.plasma.ue_para[:], NR, NZ)

            # compare if two methods give the same operatoryy
            @test A_𝐮∇ == RP.operators.A_𝐮∇

            # Compare the results
            @test isapprox(explicit_result, implicit_result, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result2, rtol=1e-10)

            # Additional test: Verify that matrix multiplication operations don't error
            @test_nowarn A_𝐮∇ * (A_𝐮∇ * RP.plasma.ue_para[:])
        end
    end
end
