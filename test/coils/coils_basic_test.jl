using Test
using RAPID2D

@testset "Coils Basic Types and Constructors" begin
    FT = Float64

    @testset "Coil Constructor and Fields" begin
        # Test basic Coil construction
        pos = (r=2.5, z=0.0)
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6

        # Test powered controllable coil
        coil_powered = Coil(pos, area, resistance, self_inductance,
                           true, true, "PF1", 1000.0, 50000.0, 100.0, 500.0)

        @test coil_powered.location.r == 2.5
        @test coil_powered.location.z == 0.0
        @test coil_powered.area == area
        @test coil_powered.resistance == resistance
        @test coil_powered.self_inductance == self_inductance
        @test coil_powered.is_powered == true
        @test coil_powered.is_controllable == true
        @test coil_powered.name == "PF1"
        @test coil_powered.max_voltage == 1000.0
        @test coil_powered.max_current == 50000.0
        @test coil_powered.current == 100.0
        @test coil_powered.voltage_ext == 500.0

        # Test passive coil
        coil_passive = Coil(pos, area, resistance, self_inductance,
                           false, false, "wall_1", nothing, nothing, -50.0, 0.0)

        @test coil_passive.is_powered == false
        @test coil_passive.is_controllable == false
        @test coil_passive.max_voltage === nothing
        @test coil_passive.max_current === nothing
        @test coil_passive.current == -50.0
        @test coil_passive.voltage_ext == 0.0

        # Test powered but not controllable coil
        coil_powered_nc = Coil(pos, area, resistance, self_inductance,
                              true, false, "heating_coil", 500.0, 10000.0)

        @test coil_powered_nc.is_powered == true
        @test coil_powered_nc.is_controllable == false
        @test coil_powered_nc.max_voltage == 500.0
    end

    @testset "Coil Constructor Validation" begin
        pos = (r=2.5, z=0.0)
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6

        # Test invalid area
        @test_throws AssertionError Coil(pos, -0.1, resistance, self_inductance,
                                         true, true, "test", 1000.0, 50000.0)

        # Test invalid resistance
        @test_throws AssertionError Coil(pos, area, -0.1, self_inductance,
                                         true, true, "test", 1000.0, 50000.0)

        # Test invalid self-inductance
        @test_throws AssertionError Coil(pos, area, resistance, -1e-6,
                                         true, true, "test", 1000.0, 50000.0)

        # Test invalid R coordinate
        invalid_position = (r=-1.0, z=0.0)
        @test_throws AssertionError Coil(invalid_position, area, resistance, self_inductance,
                                         true, true, "test", 1000.0, 50000.0)

        # Test controllable coil must be powered
        @test_throws AssertionError Coil(pos, area, resistance, self_inductance,
                                         false, true, "test", nothing, nothing)
    end
    @testset "CoilSystem Constructor" begin
        # Test empty CoilSystem
        system_empty = CoilSystem{FT}()
        @test length(system_empty.coils) == 0
        @test system_empty.n_total == 0
        @test system_empty.n_powered == 0
        @test system_empty.n_controllable == 0
        @test length(system_empty.powered_indices) == 0
        @test length(system_empty.controllable_indices) == 0
        @test length(system_empty.passive_indices) == 0
        @test system_empty.μ0 ≈ 1.25663706212e-6
        @test system_empty.cu_resistivity ≈ 1.68e-8

        # Test CoilSystem with coils
        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        coil2 = Coil((r=2.5, z=-0.5), 0.01, 0.001, 1e-6, true, false, "heating", 500.0, 10000.0)
        coil3 = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall", nothing, nothing)

        coils = [coil1, coil2, coil3]
        system = CoilSystem{FT}(coils)

        @test system.n_total == 3
        @test system.n_powered == 2
        @test system.n_controllable == 1
        @test system.powered_indices == [1, 2]
        @test system.controllable_indices == [1]
        @test system.passive_indices == [3]
    end

    @testset "Helper Functions" begin
        # Test calculate_coil_resistance
        area = π * 0.02^2
        major_radius = 2.5
        resistivity = 1.68e-8

        resistance = calculate_coil_resistance(area, major_radius, resistivity)
        expected = resistivity * 2π * major_radius / area
        @test resistance ≈ expected

        # Test calculate_self_inductance
        μ0 = 1.25663706212e-6
        self_L = calculate_self_inductance(area, major_radius, μ0)
        coil_radius = sqrt(area / π)
        expected_L = μ0 * major_radius * (log(8 * major_radius / coil_radius) - 2 + 0.25)
        @test self_L ≈ expected_L

        # Test create_coil_from_parameters
        coil = create_coil_from_parameters(2.0, 1.0, area, "test_coil", true,
                                          μ0, resistivity; is_controllable=false,
                                          max_voltage=800.0, max_current=30000.0)

        @test coil.location.r == 2.0
        @test coil.location.z == 1.0
        @test coil.area == area
        @test coil.name == "test_coil"
        @test coil.is_powered == true
        @test coil.is_controllable == false
        @test coil.max_voltage == 800.0
        @test coil.max_current == 30000.0
        @test coil.resistance ≈ calculate_coil_resistance(area, 2.0, resistivity)
        @test coil.self_inductance ≈ calculate_self_inductance(area, 2.0, μ0)
    end
end
