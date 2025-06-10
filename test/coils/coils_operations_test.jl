using Test
using RAPID2D

@testset "Coil System Operations" begin
    FT = Float64

    @testset "Adding Coils to System" begin
        system = CoilSystem{FT}()

        # Create test coils
        pf_coil = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        cs_coil = Coil((r=1.8, z=0.0), 0.015, 0.0008, 2e-6, true, false, "CS", 2000.0, 100000.0)
        wall_segment = Coil((r=2.01, z=0.2), 0.005, 0.01, 1e-7, false, false, "wall_1", nothing, nothing)

        # Test adding coils
        add_coil!(system, pf_coil)
        @test system.n_total == 1
        @test system.n_powered == 1
        @test system.n_controllable == 1
        @test system.powered_indices == [1]
        @test system.controllable_indices == [1]
        @test system.passive_indices == []

        add_coil!(system, cs_coil)
        @test system.n_total == 2
        @test system.n_powered == 2
        @test system.n_controllable == 1
        @test system.powered_indices == [1, 2]
        @test system.controllable_indices == [1]

        add_coil!(system, wall_segment)
        @test system.n_total == 3
        @test system.n_powered == 2
        @test system.n_controllable == 1
        @test system.passive_indices == [3]
    end

    @testset "Getting Coils by Type" begin
        system = CoilSystem{FT}()

        pf_coil = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        cs_coil = Coil((r=1.8, z=0.0), 0.015, 0.0008, 2e-6, true, false, "CS", 2000.0, 100000.0)
        wall1 = Coil((r=2.01, z=0.2), 0.005, 0.01, 1e-7, false, false, "wall_1", nothing, nothing)
        wall2 = Coil((r=2.01, z=-0.2), 0.005, 0.01, 1e-7, false, false, "wall_2", nothing, nothing)

        add_coil!(system, pf_coil)
        add_coil!(system, cs_coil)
        add_coil!(system, wall1)
        add_coil!(system, wall2)

        # Test get_powered_coils
        powered_coils = get_powered_coils(system)
        @test length(powered_coils) == 2
        @test powered_coils[1].name == "PF1"
        @test powered_coils[2].name == "CS"

        # Test get_controllable_coils
        controllable_coils = get_controllable_coils(system)
        @test length(controllable_coils) == 1
        @test controllable_coils[1].name == "PF1"

        # Test get_passive_coils
        passive_coils = get_passive_coils(system)
        @test length(passive_coils) == 2
        @test passive_coils[1].name == "wall_1"
        @test passive_coils[2].name == "wall_2"
    end

    @testset "Finding Coils by Name" begin
        system = CoilSystem{FT}()

        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF_Upper", 1000.0, 50000.0)
        coil2 = Coil((r=2.5, z=-0.5), 0.01, 0.001, 1e-6, true, true, "PF_Lower", 1000.0, 50000.0)

        add_coil!(system, coil1)
        add_coil!(system, coil2)

        # Test finding existing coils
        @test find_coil_by_name(system, "PF_Upper") == 1
        @test find_coil_by_name(system, "PF_Lower") == 2

        # Test finding non-existent coil
        @test find_coil_by_name(system, "NonExistent") === nothing
    end

    @testset "Individual Voltage Control" begin
        system = CoilSystem{FT}()

        pf_coil = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        wall_coil = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall", nothing, nothing)

        add_coil!(system, pf_coil)
        add_coil!(system, wall_coil)

        # Test setting voltage for powered coil
        set_coil_voltage!(system, "PF1", 750.0)
        @test get_coil_voltage(system, "PF1") == 750.0
        @test system.coils[1].voltage_ext == 750.0

        # Test getting voltage for passive coil (should be 0)
        @test get_coil_voltage(system, "wall") == 0.0

        # Test error handling for non-existent coil
        @test_throws ErrorException set_coil_voltage!(system, "NonExistent", 100.0)
        @test_throws ErrorException get_coil_voltage(system, "NonExistent")

        # Test error handling for setting voltage on passive coil
        @test_throws ErrorException set_coil_voltage!(system, "wall", 100.0)
    end

    @testset "Individual Current Control" begin
        system = CoilSystem{FT}()

        pf_coil = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        wall_coil = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall", nothing, nothing)

        add_coil!(system, pf_coil)
        add_coil!(system, wall_coil)

        # Test setting current for any coil
        set_coil_current!(system, "PF1", 25000.0)
        @test get_coil_current(system, "PF1") == 25000.0

        set_coil_current!(system, "wall", -500.0)  # Induced current
        @test get_coil_current(system, "wall") == -500.0

        # Test error handling for non-existent coil
        @test_throws ErrorException set_coil_current!(system, "NonExistent", 100.0)
        @test_throws ErrorException get_coil_current(system, "NonExistent")
    end

    @testset "Position Access" begin
        system = CoilSystem{FT}()

        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "coil1", 1000.0, 50000.0)
        coil2 = Coil((r=1.8, z=-0.3), 0.015, 0.0008, 2e-6, false, false, "coil2", nothing, nothing)

        add_coil!(system, coil1)
        add_coil!(system, coil2)

        # Test get_coil_positions
        R_coords, Z_coords = get_coil_positions(system)
        @test R_coords == [2.5, 1.8]
        @test Z_coords == [0.5, -0.3]

        # Test get_powered_coil_positions
        R_powered, Z_powered = get_powered_coil_positions(system)
        @test R_powered == [2.5]
        @test Z_powered == [0.5]
    end
end
