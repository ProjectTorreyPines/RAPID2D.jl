using RAPID2D
using Test
using LinearAlgebra
using Statistics

# Test basic physics module functionality
@testset "Physics Module Basic Tests" begin
    # Create a simple test configuration
    config = SimulationConfig{Float64}(
        device_Name = "manual",         # Manual configuration
        NR = 40,                       # Grid points in R direction
        NZ = 80,                       # Grid points in Z direction
        prefilled_gas_pressure = 1.0e-2, # Pressure in Pa
        R0B0 = 1.0,                    # R0B0 value for toroidal field
        dt = 1.0e-8,             # Fixed timestep
        snap1D_Interval_s = 1.0e-7,
        snap2D_Interval_s = 1.0e-6
    )

    # Create RAPID instance
	RP = create_rapid_object(;config)

    # Create a RAPID object directly from the configuration
    RP = RAPID{Float64}(config)
    initialize!(RP)

    # Test 1: Check if electron and ion densities were properly initialized
    @test all(RP.plasma.ne[RP.G.nodes.in_wall_nids] .== 1.0e6)
    @test all(RP.plasma.ne[RP.G.nodes.out_wall_nids] .== 0.0)
    @test RP.plasma.ne == RP.plasma.ni # Ion density should match electron density

    # Test 2: Check if temperature was properly initialized
    @test all(RP.plasma.Te_eV .≈ RP.config.constants.room_T_eV)
    @test all(RP.plasma.Ti_eV .≈ RP.config.constants.room_T_eV)

    # Test 3: Check magnetic field and unit vectors
    @test all(isapprox.(RP.fields.bR.^2 + RP.fields.bZ.^2 + RP.fields.bϕ.^2, 1.0, atol=1.0e-10))
    @test all(RP.fields.BR .== 0.0) # Default manual setup has BR = 0
    @test all(RP.fields.BZ .> 0.0)  # Default manual setup has BZ > 0

    # Test 4: Check electric field calculation
    @test all(RP.fields.E_para_tot .== RP.fields.E_para_ext)
    @test all(RP.fields.E_para_ext .== RP.fields.Eϕ_ext .* RP.fields.bϕ)

    # Test 5: Check velocity initialization
    @test all(RP.plasma.ue_para .== 0.0)
    @test all(RP.plasma.ui_para .== 0.0)
end

# Test the reaction rate coefficient functions
@testset "Reaction Rate Coefficient Tests" begin
    # Create a simple test configuration
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 10, NZ = 10,
        prefilled_gas_pressure = 1.0e-2,
        R0B0 = 1.0
    )

    # Create RAPID instance
    RP = RAPID{Float64}(config)

    # Initialize the simulation
    initialize!(RP)

    # Test electron reaction rate coefficient function
    RRC_iz = get_electron_RRC(RP, RP.eRRCs, :Ionization)
    RRC_mom = get_electron_RRC(RP, RP.eRRCs, :Momentum)

    # Verify shapes
    @test size(RRC_iz) == (RP.G.NR, RP.G.NZ)
    @test size(RRC_mom) == (RP.G.NR, RP.G.NZ)

    # Values should be positive (for non-zero density regions)
    @test all(RRC_iz .>= 0.0)
    @test all(RRC_mom .>= 0.0)

    # Test ion reaction rate coefficient function
    iRRC_elastic = get_H2_ion_RRC(RP, RP.iRRCs, :Elastic)
    iRRC_cx = get_H2_ion_RRC(RP, RP.iRRCs, :Charge_Exchange)

    # Verify shapes
    @test size(iRRC_elastic) == (RP.G.NR, RP.G.NZ)
    @test size(iRRC_cx) == (RP.G.NR, RP.G.NZ)

    # Values should be positive
    @test all(iRRC_elastic .>= 0.0)
    @test all(iRRC_cx .>= 0.0)
end

# Test the density source, diffusion, and convection terms
@testset "Density Transport RHS Terms Tests" begin
    # Create a simple test configuration
    config = SimulationConfig{Float64}(
        device_Name = "manual",
        NR = 20, NZ = 20,
        prefilled_gas_pressure = 5e-3,
        R0B0 = 1.0,
        dt = 1.0e-6,
        Dperp0 = 0.1,
    )

    # Create RAPID instance with flags enabled
    RP = RAPID{Float64}(config)
    RP.flags.Implicit = false  # Use explicit scheme for simplicity
    RP.flags.diffu = true      # Enable diffusion
    RP.flags.convec = true     # Enable convection
    RP.flags.src = true        # Enable sources
	RP.flags.Include_ud_convec_term = false
    # RP.flags.Ionz_method = "Townsend_coeff"
    RP.flags.Ionz_method = "Xsec"
	# -6.50391e5
    # Initialize
    initialize!(RP)

    # Setup a test case with non-zero parallel velocity
	RP.plasma.Te_eV .= 10.0 # Set initial electron temperature

	for _ in 1:100
		update_ue_para!(RP)
	end

	@test mean(RP.plasma.ue_para[RP.G.nodes.in_wall_nids]) ≈ -382634.21437302034

    # Calculate source terms
    calculate_density_source_terms!(RP)

    # Test source term calculation
    @test !all(RP.operators.neRHS_src .== 0.0)  # Should have non-zero source terms
    @test all(RP.operators.neRHS_src[RP.G.nodes.out_wall_nids] .== 0.0)  # Zero outside wall

    # Calculate diffusion terms
    calculate_density_diffusion_terms!(RP)

    # Test diffusion term calculation (will be zero initially since ne is uniform inside wall)
    @test all(RP.operators.neRHS_diffu[RP.G.nodes.in_wall_nids] .== 0.0)

    # Modify density to create gradients
    inside_idx = RP.G.nodes.in_wall_nids
    center = [RP.G.NR ÷ 2, RP.G.NZ ÷ 2]
    for i in inside_idx
        r, z = RP.G.nodes.rid[i], RP.G.nodes.zid[i]
        dist = sqrt((r - center[1])^2 + (z - center[2])^2)
        RP.plasma.ne[i] = 1.0e6 * exp(-dist^2 / 20.0)
    end

    # Recalculate diffusion terms with non-uniform density
    calculate_density_diffusion_terms!(RP)

    # Now diffusion should be non-zero inside wall
    @test !all(RP.operators.neRHS_diffu[inside_idx] .== 0.0)

    # Calculate convection terms
    # Set up a test velocity field
    RP.plasma.ueR .= 1000.0
    RP.plasma.ueZ .= 0.0

    calculate_density_convection_terms!(RP)

    # Test convection term calculation
    @test !all(RP.operators.neRHS_convec[inside_idx] .== 0.0)
    @test all(RP.operators.neRHS_convec[RP.G.nodes.out_wall_nids] .== 0.0)
end