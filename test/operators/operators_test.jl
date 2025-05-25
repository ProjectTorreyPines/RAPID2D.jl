using RAPID2D
using LinearAlgebra
using SparseArrays
using Test


@testset "Basic differential operators" begin
    # Define test parameters
    NR, NZ = 50, 100  # Small grid for testing
    FT = Float64    # Floating point type

    # Create a proper RAPID object for testing using create_rapid_object
    config = RAPID2D.SimulationConfig{FT}(
        prefilled_gas_pressure=4e-3,
        R0B0=1.5*1.8,
        NR=NR,
        NZ=NZ
    )
    # Create the RAPID object
    RP = create_rapid_object(; config=config)


    FR  = zeros(FT, NR, NZ)
    FZ  = zeros(FT, NR, NZ)
    # Create a vortex-like flow vector field
    for j in 1:NZ, i in 1:NR
        # Centered coordinates
        x = i - NR/2
        y = j - NZ/2
        r = sqrt(x^2 + y^2)

        if r > 0
            # Circular flow around center
            FR[i,j] = -0.1 * y / r
            FZ[i,j] = 0.1 * x / r
        else
            FR[i,j] = 0.0
            FZ[i,j] = 0.0
        end
    end

    OP = RP.operators

	# Explicit method
    div_numerical_1 = calculate_divergence(RP.G, FR, FZ) # Using central differencing

    # using operators matrix-vector multiplication
    div_numerical_2 = OP.𝐽⁻¹∂R_𝐽 * FR .+ OP.∂Z * FZ

    @test isapprox(div_numerical_1, div_numerical_2, rtol=1e-14)

    # Test convient dispatches
    @test div_numerical_2 == reshape(calculate_divergence(OP, FR[:], FZ[:]), NR, NZ)
    @test div_numerical_2 == calculate_divergence(OP, FR, FZ)

    @testset "Convenient dispatches" begin
        ∂R = RAPID2D.construct_∂R_operator(RP)
        𝐽⁻¹∂R_𝐽 = construct_𝐽⁻¹∂R_𝐽_operator(RP)
        ∂Z = construct_∂Z_operator(RP)

        @test ∂R == OP.∂R
        @test 𝐽⁻¹∂R_𝐽 == OP.𝐽⁻¹∂R_𝐽
        @test ∂Z == OP.∂Z
    end

    # Analytical test cases
    @testset "Analytical divergence tests" begin
        # Get grid coordinates
        R2D = RP.G.R2D
        Z2D = RP.G.Z2D
        interior_points = 2:NR-1, 2:NZ-1

        # Test Case 1: Radial Flow (FR = R, FZ = 0)
        let
            k = 1.0  # Constant
            FR = k .* R2D
            FZ = zeros(FT, NR, NZ)

            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing
            div_analytical = fill(2*k, (NR, NZ))

            @test isapprox(div_numerical[interior_points...], div_analytical[interior_points...], rtol=1e-4)
        end

        # Test Case 2: Linear Flow (FR = a·R, FZ = b·Z)
        let
            a = 1.0
            b = 2.0
            FR = a .* R2D
            FZ = b .* Z2D

            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing
            div_analytical = fill(2*a + b, (NR, NZ))

            @test isapprox(div_numerical[interior_points...], div_analytical[interior_points...], rtol=1e-2)
        end

        # Test Case 3 (divergence-free): (FR = R²-Z², FZ = -3*R*Z + Z³/(3*R))
        let
            FR = @. R2D^2 - Z2D^2
            FZ = @. -3 * R2D * Z2D + Z2D^3/(3*R2D)
            div_numerical = RAPID2D.calculate_divergence(RP.G, FR, FZ)  # Using central differencing

            @test maximum(abs.(div_numerical[interior_points...])) < 1e-2
        end
    end
end

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

    explicit_result2 = compute_∇𝐃∇f_directly(RP, test_density)

    # Implicit method
    ∇𝐃∇ = RAPID2D.construct_∇𝐃∇_operator(RP)
    implicit_result = ∇𝐃∇ * test_density

    # Calculate implicit diffusion using RAPID2D's internal way that can update the operator more efficiently
    # This is useful for large simulations where we want to avoid re-creating the operator
    RP.operators.∇𝐃∇ = RAPID2D.construct_∇𝐃∇_operator(RP)
    RAPID2D.update_∇𝐃∇_operator!(RP)
    implicit_result2 = RP.operators.∇𝐃∇ * test_density

    # compare if two methods give the same operatoryy
    @test ∇𝐃∇ == RP.operators.∇𝐃∇
    @test !(∇𝐃∇ === RP.operators.∇𝐃∇)

	# Comparison
	@test isapprox(explicit_result, explicit_result2, rtol=1e-10)
	@test isapprox(explicit_result, implicit_result, rtol=1e-10)
	@test isapprox(explicit_result, implicit_result2, rtol=1e-10)


    # Additional test: Verify that matrix multiplication operations don't error
    # This is to ensure our implementation works with typical linear algebra operations
    @test_nowarn ∇𝐃∇ * (∇𝐃∇ * test_density)
end

@testset "Convection operator [-∇⋅(n 𝐮)]  - Explicit vs Implicit" begin
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

            explicit_result2 = -compute_∇f𝐮_directly(RP, test_density, uR, uZ; flag_upwind)

            # Calculate implicit convection term
            ∇𝐮 = RAPID2D.construct_∇𝐮_operator(RP, uR, uZ; flag_upwind)
            implicit_result = -∇𝐮 * test_density

            # Calculate implicit convection using RAPID2D's internal way that can update the operator more efficiently
            # This is useful for large simulations where we want to avoid re-creating the operator
            RP.operators.∇𝐮 = RAPID2D.construct_∇𝐮_operator(RP; flag_upwind)
            implicit_result2 = -RP.operators.∇𝐮 * test_density

            # compare if two methods give the same operatoryy
            @test ∇𝐮 == RP.operators.∇𝐮

            # Compare the results
            @test isapprox(explicit_result, explicit_result2, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result2, rtol=1e-10)

            # Additional test: Verify that matrix multiplication operations don't error
            result1 = ∇𝐮 * (∇𝐮 * test_density)
            result2 = (∇𝐮 * ∇𝐮) * test_density
            result3 = (∇𝐮^2) * test_density

            @test isapprox(result1, result2, rtol=1e-10)
            @test isapprox(result1, result3, rtol=1e-10)
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

        for flag_upwind in [true, false]
            RAPID2D.calculate_ne_convection_explicit_RHS!(RP, test_density, uR, uZ; flag_upwind)
            explicit_result = copy(RP.operators.neRHS_convec)

            ∇𝐮 = RAPID2D.construct_∇𝐮_operator(RP, uR, uZ; flag_upwind)
            implicit_result = -∇𝐮 * test_density

            RP.operators.∇𝐮 = RAPID2D.construct_∇𝐮_operator(RP; flag_upwind)
            implicit_result2 = -RP.operators.∇𝐮 * test_density

            @test isapprox(explicit_result, implicit_result, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result2, rtol=1e-10)
        end
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

            explicit_result2 = compute_𝐮∇f_directly(RP, RP.plasma.ue_para; flag_upwind)

            # Calculate implicit convection term
            𝐮∇ = RAPID2D.construct_𝐮∇_operator(RP; flag_upwind)
            implicit_result = reshape(𝐮∇ * RP.plasma.ue_para[:], NR, NZ)

            # Calculate implicit convection using RAPID2D's internal way that can update the operator more efficiently
            # This is useful for large simulations where we want to avoid re-creating the operator
            RP.operators.𝐮∇ = RAPID2D.construct_𝐮∇_operator(RP; flag_upwind)
            implicit_result2 = reshape(RP.operators.𝐮∇ * RP.plasma.ue_para[:], NR, NZ)

            # compare if two methods give the same operatoryy
            @test 𝐮∇ == RP.operators.𝐮∇

            # Compare the results
            @test isapprox(explicit_result, explicit_result2, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result, rtol=1e-10)
            @test isapprox(explicit_result, implicit_result2, rtol=1e-10)

            # Additional test: Verify that matrix multiplication operations don't error
            @test_nowarn 𝐮∇ * (𝐮∇ * RP.plasma.ue_para[:])
        end
    end
end
