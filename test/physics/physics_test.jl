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
end

@testset "Pure Convection Test: with constant ue_para" begin
    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=50, NZ=70,
        R_min=0.1, R_max=0.5,
        Z_min=-0.4, Z_max=0.4,
        dt=1e-6, t_end_s=100e-6,
        R0B0=1.0,
        Dpara0=10.0, Dperp0=0.1,
        prefilled_gas_pressure=5e-3,
        wall_R=[0.15, 0.45, 0.45, 0.15],
        wall_Z=[-0.35, -0.35, 0.35, 0.35]
    )

    # Create RAPID object
    RP = RAPID{FT}(config)

    RP.flags = SimulationFlags(
        convec=true,           # Enable convection (simplify initial test)
        # Disable unnecessary flags for this test
        src=false,
        diffu=false,
        ud_evolve=false,
        ud_method="Xsec",
        Te_evolve=false,
        Ti_evolve=false,
        Ampere=false,
        E_para_self_ES=false,
        E_para_self_EM=false,
        Gas_evolve=false,
        update_ni_independently=false,
        Include_ud_convec_term=false,
        Coulomb_Collision=false,
        negative_n_correction=false
    )

    # Initial conditions: Gaussian electron density distribution centered in domain
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = (config.R_max - config.R_min) / 16
    sigma_Z = (config.Z_max - config.Z_min) / 16
    peak_density = 1.0e6  # Peak density [m^-3]

    initialize!(RP)
    # Initialize electron density
    ini_ne = zeros(FT, RP.G.NR, RP.G.NZ)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R = RP.G.R2D[i, j]
        Z = RP.G.Z2D[i, j]
        # Gaussian density profile
        ini_ne[i, j] = peak_density * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    ini_ne[RP.G.nodes.out_wall_nids] .= 0.0

    ini_ue_para = 1e6 # Initial drift velocity [m/s]
    ini_BR_ext = 10e-4
    ini_BZ_ext = 20e-4;

    function _set_initial_conditions!(RP)
        # Set up initial conditions
        RP.plasma.ne = copy(ini_ne)
        RP.plasma.ue_para .= copy(ini_ue_para) # Initial drift velocity [m/s]
        RP.fields.BR_ext .= copy(ini_BR_ext)
        RP.fields.BZ_ext .= copy(ini_BZ_ext)
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    for flag_implicit in [false, true], flag_upwind in [false, true]
        RP.flags.Implicit = flag_implicit
        RP.flags.upwind = flag_upwind

        initialize!(RP)
        _set_initial_conditions!(RP)

        # Check initial state
        @test all(RP.plasma.ne .>= 0.0)

        # expect the displacement by the constant ue_para
        ΔR_2D = @. RP.plasma.ue_para*RP.fields.bR*RP.t_end_s
        ΔZ_2D = @. RP.plasma.ue_para*RP.fields.bZ*RP.t_end_s
        expected_R_displacement = sum(ini_ne.*ΔR_2D)/sum(ini_ne)
        expected_Z_displacement = sum(ini_ne.*ΔZ_2D)/sum(ini_ne)

        # Run simulation
        println("")
        println("")
        @info "Starting simulation with (implicit=$flag_implicit, upwind=$flag_upwind)..."
        @time RAPID2D.run_simulation!(RP);

        # Check the actual displacement
        actual_R_displacement = sum(RP.plasma.ne.*RP.G.R2D)/sum(RP.plasma.ne) - R0
        actual_Z_displacement = sum(RP.plasma.ne.*RP.G.Z2D)/sum(RP.plasma.ne) - Z0

        println("Actual (R,Z) displacements: (", actual_R_displacement, ", ", actual_Z_displacement,")")
        println("minimum ne: ", minimum(RP.plasma.ne))

        # Check final state
        if flag_upwind
            @test all(RP.plasma.ne .>= 0.0)  # No negative densities
        else
            @test all(RP.plasma.ne .>= -1e-9*maximum(ini_ne))  # No significant negative densities
        end

        @test isapprox(actual_R_displacement, expected_R_displacement, rtol=5e-2)
        @test isapprox(actual_Z_displacement, expected_Z_displacement, rtol=5e-2)
    end

    # explciit
    RP.flags.Implicit = false
    initialize!(RP)
    _set_initial_conditions!(RP)
    RAPID2D.run_simulation!(RP);
    RP_explicit = deepcopy(RP);

    # implicit with θ=0
    RP.flags.Implicit = true
    RP.flags.Implicit_weight = 0.0
    initialize!(RP)
    _set_initial_conditions!(RP)
    RAPID2D.run_simulation!(RP);
    RP_implicit_0 = deepcopy(RP);

    # implicit with θ=1
    RP.flags.Implicit = true
    RP.flags.Implicit_weight = 1.0
    initialize!(RP)
    _set_initial_conditions!(RP)
    RAPID2D.run_simulation!(RP);
    RP_implicit_1 = deepcopy(RP);

    # implicit with θ=0 must be the same as explicit results
    @test isapprox(RP_explicit.plasma.ne, RP_implicit_0.plasma.ne, rtol=1e-12)

    # implicit with θ=1 could be similar to the explicit results but not the same
    @test isapprox(RP_explicit.plasma.ne, RP_implicit_1.plasma.ne, rtol=5e-2) # similar (within 5% error)
    @test !isapprox(RP_explicit.plasma.ne, RP_implicit_1.plasma.ne, rtol=1e-12) # not the same
end

@testset "Pure Diffusion Test" begin
    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=50, NZ=70,
        R_min=0.8, R_max=2.2,
        Z_min=-1.2, Z_max=1.2,
        dt=1e-6, t_end_s=100e-6,
        R0B0=1.0,
        Dpara0=0.0, Dperp0=100.0,
        prefilled_gas_pressure=5e-3,
        wall_R=[1.0, 2.0, 2.0, 1.0],
        wall_Z=[-1.0, -1.0, 1.0, 1.0]
    )

    # Create RAPID object
    RP = RAPID{FT}(config)

    RP.flags = SimulationFlags(
        diffu=true,           # Enable convection (simplify initial test)
        # Disable unnecessary flags for this test
        src=false,
        convec=false,
        ud_evolve=false,
        ud_method="Xsec",
        Te_evolve=false,
        Ti_evolve=false,
        Ampere=false,
        E_para_self_ES=false,
        E_para_self_EM=false,
        Gas_evolve=false,
        update_ni_independently=false,
        Include_ud_convec_term=false,
        Coulomb_Collision=false,
        negative_n_correction=false
    )

    # Initial conditions: Gaussian electron density distribution centered in domain
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.1
    sigma_Z = 0.1
    peak_density = 1.0e6  # Peak density [m^-3]

    initialize!(RP)
    # Initialize electron density
    ini_ne = zeros(FT, RP.G.NR, RP.G.NZ)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R = RP.G.R2D[i, j]
        Z = RP.G.Z2D[i, j]
        # Gaussian density profile
        ini_ne[i, j] = peak_density * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    ini_ne[RP.G.nodes.out_wall_nids] .= 0.0

    function _set_initial_conditions!(RP, ini_ne, ini_BR_ext, ini_BZ_ext)
        RP.plasma.ne = copy(ini_ne)
        RP.fields.BR_ext .= copy(ini_BR_ext)
        RP.fields.BZ_ext .= copy(ini_BZ_ext)
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    function _measure_σR_and_σZ_of_ne(ne)
        return (σR = sqrt(sum(ne.*(RP.G.R2D.-R0).^2)/sum(ne)),
                σZ = sqrt(sum(ne.*(RP.G.Z2D.-Z0).^2)/sum(ne)))
    end

    # pure Dperp0
    for RP.flags.Implicit in [false, true]
        RP.config.Dpara0 = 0
        RP.config.Dperp0 = 100

        initialize!(RP)
        _set_initial_conditions!(RP, ini_ne, 0.0, 0.0)
        RAPID2D.run_simulation!(RP);

        σR0, σZ0 =_measure_σR_and_σZ_of_ne(ini_ne)
        σR_end, σZ_end =_measure_σR_and_σZ_of_ne(RP.plasma.ne)

        mean_σ0 = (σR0 + σZ0) / 2
        mean_σ_end = (σR_end + σZ_end) / 2

        estimated_Dperp0 = (mean_σ_end^2 - mean_σ0^2)/(2.0*RP.time_s)
        @test isapprox(estimated_Dperp0, RP.transport.Dperp0; rtol=0.05) # 5% error
    end

    # pure Dpara0
    for RP.flags.Implicit in [false, true]
        RP.config.Dpara0 = 1e6
        RP.config.Dperp0 = 0

        initialize!(RP)
        _set_initial_conditions!(RP, ini_ne, 50e-4, 100e-4)
        RAPID2D.run_simulation!(RP);

        σR0, σZ0 =_measure_σR_and_σZ_of_ne(ini_ne)
        σR_end, σZ_end =_measure_σR_and_σZ_of_ne(RP.plasma.ne)

        avg_DRR = sum(RP.transport.DRR.*ini_ne)/sum(ini_ne)
        avg_DZZ = sum(RP.transport.DZZ.*ini_ne)/sum(ini_ne)

        estimated_DRR = (σR_end^2 - σR0^2)/(2.0*RP.time_s)
        estimated_DZZ = (σZ_end^2 - σZ0^2)/(2.0*RP.time_s)

        @test isapprox(avg_DRR, estimated_DRR; rtol=0.02) # 2% error
        @test isapprox(avg_DZZ, estimated_DZZ; rtol=0.02) # 2% error
    end
end

@testset "Free Accel & Heating Test: no collision" begin
    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=50, NZ=70,
        R_min=0.8, R_max=2.2,
        Z_min=-1.2, Z_max=1.2,
        dt=1e-6, t_end_s=100e-6,
        R0B0=1.0,
        Dpara0=0.0, Dperp0=0.0,
        prefilled_gas_pressure=5e-3,
        wall_R=[1.0, 2.0, 2.0, 1.0],
        wall_Z=[-1.0, -1.0, 1.0, 1.0]
    )

    # Create RAPID object
    RP = RAPID{FT}(config)

    RP.flags = SimulationFlags(
        ud_evolve=true,
        ud_method="Xsec",
        Te_evolve=false,
        Ti_evolve=false,
        # Disable unnecessary flags for this test
        src=false, diffu=false, convec=false, Ampere=false,
        E_para_self_ES=false, E_para_self_EM=false, Gas_evolve=false,
        update_ni_independently=false, Include_ud_convec_term=false,
        Coulomb_Collision=false, negative_n_correction=false
    )

    # Initial conditions: Gaussian electron density distribution centered in domain
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.1
    sigma_Z = 0.1
    peak_density = 1.0e6  # Peak density [m^-3]

    initialize!(RP)
    # Initialize electron density
    ini_n = zeros(FT, RP.G.NR, RP.G.NZ)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
        ini_n[i, j] = peak_density * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    ini_n[RP.G.nodes.out_wall_nids] .= 0.0

    function _set_initial_conditions!(RP, ini_n, ini_BR_ext, ini_BZ_ext)
        RP.plasma.ne = copy(ini_n)
        RP.plasma.ni = copy(ini_n)
        RP.fields.BR_ext .= copy(ini_BR_ext)
        RP.fields.BZ_ext .= copy(ini_BZ_ext)
        RAPID2D.combine_external_and_self_fields!(RP)
    end

    # Free acceleration
    for RP.flags.Implicit in [false, true]
        RP.flags.Atomic_Collision = false
        RP.flags.Include_ud_diffu_term = false
        initialize!(RP)
        _set_initial_conditions!(RP, ini_n, 1e-4, 1e-4)
        # _set_initial_conditions!(RP, ini_ne, 0.0, 0.0)

        RAPID2D.run_simulation!(RP);

        # Expected ue_para by free acceleration
        cnst = RP.config.constants
        ee = cnst.ee
        me = cnst.me
        mi = cnst.mi

        elec_accel_2D = @. -ee*RP.fields.E_para_ext/me
        avg_elec_accel = sum(@. ini_n*elec_accel_2D)/sum(ini_n)
        expected_avg_ue_para =avg_elec_accel*RP.config.t_end_s

        ion_accel_2D = @. ee*RP.fields.E_para_ext/mi
        avg_ion_accel = sum(@. ini_n*ion_accel_2D)/sum(ini_n)
        expected_avg_ui_para =avg_ion_accel*RP.config.t_end_s

        # Check the actual ue_para and ui_para
        actual_avg_ue_para = sum(RP.plasma.ne.*RP.plasma.ue_para)/sum(RP.plasma.ne)
        @test isapprox(actual_avg_ue_para, expected_avg_ue_para; rtol=0.01) # 1% error
        actual_avg_ui_para = sum(RP.plasma.ni.*RP.plasma.ui_para)/sum(RP.plasma.ni)
        @test isapprox(actual_avg_ui_para, expected_avg_ui_para; rtol=0.01) # 1% error
    end

end

@testset "Ionization Test without any transport" begin
    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=50, NZ=70,
        R_min=0.8, R_max=2.2,
        Z_min=-1.2, Z_max=1.2,
        dt=1e-6, t_end_s=100e-6,
        R0B0=1.0,
        Dpara0=0.0, Dperp0=0.0,
        prefilled_gas_pressure=5e-3,
        wall_R=[1.0, 2.0, 2.0, 1.0],
        wall_Z=[-1.0, -1.0, 1.0, 1.0]
    )

    # Create RAPID object
    RP = RAPID{FT}(config)

    RP.flags = SimulationFlags(
        src = true,
        # Disable other flags for this test
        ud_evolve=false,
        ud_method="Xsec",
        Te_evolve=false,
        Ti_evolve=false,
        # Disable unnecessary flags for this test
        diffu=false, convec=false, Ampere=false,
        E_para_self_ES=false, E_para_self_EM=false, Gas_evolve=false,
        update_ni_independently=false, Include_ud_convec_term=false,
        Coulomb_Collision=false, negative_n_correction=false
    )

    # Initial conditions: Gaussian electron density distribution centered in domain
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.1
    sigma_Z = 0.1
    peak_density = 1.0e6  # Peak density [m^-3]

    initialize!(RP)
    # Initialize electron density
    ini_n = zeros(FT, RP.G.NR, RP.G.NZ)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
        ini_n[i, j] = peak_density * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    ini_n[RP.G.nodes.out_wall_nids] .= 0.0

    function _set_initial_conditions!(RP, ini_n, ini_BR_ext, ini_BZ_ext)
        RP.plasma.ne = copy(ini_n)
        RP.plasma.ni = copy(ini_n)
        RP.fields.BR_ext .= copy(ini_BR_ext)
        RP.fields.BZ_ext .= copy(ini_BZ_ext)
        RAPID2D.combine_external_and_self_fields!(RP)
    end


    RP.flags.Atomic_Collision = true
    RP.flags.Include_ud_diffu_term = false
    RP.flags.Ionz_method = "Xsec"

    # Very low Te (0.1eV) => No ionization
    for RP.flags.Implicit in [false, true]
        initialize!(RP)
        _set_initial_conditions!(RP, ini_n, 1e-4, 1e-4)
        RP.plasma.Te_eV .= 0.1
        RAPID2D.run_simulation!(RP);
        @test all(RP.plasma.eGrowth_rate .== 0.0)
        @test all(RP.operators.neRHS_src .== 0.0)
        @test ini_n == RP.plasma.ne
    end

    # Sufficient Te (10 eV) => Ionization => density growth
    # Explicit method
    RP.flags.Implicit = false
    initialize!(RP)
    _set_initial_conditions!(RP, ini_n, 1e-4, 1e-4)
    RP.plasma.Te_eV .= 10.0
    RAPID2D.run_simulation!(RP);
    explicit_plasma = deepcopy(RP.plasma)

    # Implicit method (θ=0)
    RP.flags.Implicit = true
    RP.flags.Implicit_weight = 0.0
    initialize!(RP)
    _set_initial_conditions!(RP, ini_n, 1e-4, 1e-4)
    RP.plasma.Te_eV .= 10.0
    RAPID2D.run_simulation!(RP);
    implicit_plasma_zeroθ = deepcopy(RP.plasma)

    # Implicit method (θ=1)
    RP.flags.Implicit = true
    RP.flags.Implicit_weight = 1.0
    initialize!(RP)
    _set_initial_conditions!(RP, ini_n, 1e-4, 1e-4)
    RP.plasma.Te_eV .= 10.0
    RAPID2D.run_simulation!(RP);
    implicit_plasma_oneθ = deepcopy(RP.plasma)

    @test isequal(explicit_plasma.ne, implicit_plasma_zeroθ.ne) # exactly the same
    @test isapprox(explicit_plasma.ne, implicit_plasma_oneθ.ne, rtol=1e-3)
end