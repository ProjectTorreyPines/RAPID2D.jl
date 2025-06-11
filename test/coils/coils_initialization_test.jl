using Test
using RAPID2D

@testset "Coil Initialization and Vectorized Operations" begin
    FT = Float64

    @testset "Four Wall System Initialization" begin
        system = CoilSystem{FT}()

        # Test with valid n_total (divisible by 4)
        initialize_four_wall_system!(system, 20)

        @test system.n_total == 20
        @test system.n_powered == 0  # All wall segments are passive
        @test system.n_controllable == 0
        @test length(system.passive_indices) == 20
        @test length(system.powered_indices) == 0
        @test length(system.controllable_indices) == 0

        # Check that all coils are passive wall segments
        for coil in system.coils
            @test coil.is_powered == false
            @test coil.is_controllable == false
            @test coil.max_voltage === nothing
            @test coil.max_current === nothing
            @test startswith(coil.name, "wall_")
        end

        # Test invalid n_total (not divisible by 4)
        system_invalid = CoilSystem{FT}()
        @test_throws ErrorException initialize_four_wall_system!(system_invalid, 15)
    end

    @testset "Single Wall System Initialization" begin
        system = CoilSystem{FT}()

        initialize_single_wall_system!(system, 10)

        @test system.n_total == 10
        @test system.n_powered == 0
        @test system.n_controllable == 0
        @test length(system.passive_indices) == 10

        # Check that all coils are outer wall segments
        for coil in system.coils
            @test coil.is_powered == false
            @test coil.is_controllable == false
            @test coil.location.r ≈ 2.01  # Outer wall radius
            @test startswith(coil.name, "wall_outer_")
        end
    end

    @testset "Add Control Coils" begin
        system = CoilSystem{FT}()

        # Define control coil specifications
        control_specs = [
            (r=2.5, z=0.5, area=π*0.02^2, name="PF_Upper", max_voltage=1000.0, max_current=50000.0),
            (r=2.5, z=0.0, area=π*0.03^2, name="CS_Main", max_voltage=2000.0, max_current=100000.0),
            (r=2.5, z=-0.5, area=π*0.02^2, name="PF_Lower", max_voltage=1000.0, is_controllable=false),
            (r=1.8, z=0.0, area=π*0.015^2, name="OH_Heating", max_voltage=500.0, is_controllable=false)
        ]

        add_control_coils!(system, control_specs)

        @test system.n_total == 4
        @test system.n_powered == 4
        @test system.n_controllable == 2  # First two are controllable by default, last two explicitly set to false

        # Check individual coils
        @test system.coils[1].name == "PF_Upper"
        @test system.coils[1].is_controllable == true
        @test system.coils[2].name == "CS_Main"
        @test system.coils[2].is_controllable == true
        @test system.coils[3].name == "PF_Lower"
        @test system.coils[3].is_controllable == false
        @test system.coils[4].name == "OH_Heating"
        @test system.coils[4].is_controllable == false
    end

    @testset "Example Tokamak Initialization" begin
        system = CoilSystem{FT}()

        initialize_example_tokamak_coils!(system)

        # Should have wall segments + control coils
        @test system.n_total > 40  # At least 40 wall segments plus control coils
        @test system.n_powered > 0  # Should have some control coils
        @test system.n_controllable > 0  # Control coils should be controllable
        @test length(system.passive_indices) >= 40  # Wall segments

        # Check that we have the expected control coils
        control_coil_names = [coil.name for coil in get_powered_coils(system)]
        expected_names = ["PF1", "CS", "PF2", "OH"]
        for name in expected_names
            @test name in control_coil_names
        end
    end

    @testset "Vectorized Current Operations" begin
        system = CoilSystem{FT}()

        # Add test coils
        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0, 100.0)
        coil2 = Coil((r=2.5, z=-0.5), 0.01, 0.001, 1e-6, true, false, "PF2", 1000.0, 50000.0, 200.0)
        coil3 = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall", nothing, nothing, -50.0)

        add_coil!(system, coil1)
        add_coil!(system, coil2)
        add_coil!(system, coil3)

        # Test get_all_currents
        all_currents = get_all_currents(system)
        @test all_currents == [100.0, 200.0, -50.0]

        # Test get_powered_currents
        powered_currents = get_powered_currents(system)
        @test powered_currents == [100.0, 200.0]

        # Test get_controllable_currents
        controllable_currents = get_controllable_currents(system)
        @test controllable_currents == [100.0]

        # Test set_all_currents!
        new_all_currents = [500.0, 600.0, -100.0]
        set_all_currents!(system, new_all_currents)
        @test get_all_currents(system) == new_all_currents

        # Test set_powered_currents!
        new_powered_currents = [1000.0, 1200.0]
        set_powered_currents!(system, new_powered_currents)
        @test get_powered_currents(system) == new_powered_currents
        @test system.coils[3].current == -100.0  # Passive coil unchanged

        # Test set_controllable_currents!
        new_controllable_currents = [800.0]
        set_controllable_currents!(system, new_controllable_currents)
        @test get_controllable_currents(system) == new_controllable_currents
        @test system.coils[2].current == 1200.0  # Non-controllable powered coil unchanged

        # Test error handling for wrong vector sizes
        @test_throws ErrorException set_all_currents!(system, [1.0, 2.0])  # Wrong size
        @test_throws ErrorException set_powered_currents!(system, [1.0])   # Wrong size
        @test_throws ErrorException set_controllable_currents!(system, [1.0, 2.0])  # Wrong size
    end

    @testset "Vectorized Voltage Operations" begin
        system = CoilSystem{FT}()

        # Add test coils
        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0, 0.0, 300.0)
        coil2 = Coil((r=2.5, z=-0.5), 0.01, 0.001, 1e-6, true, false, "PF2", 1000.0, 50000.0, 0.0, 400.0)
        coil3 = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall", nothing, nothing, 0.0, 0.0)

        add_coil!(system, coil1)
        add_coil!(system, coil2)
        add_coil!(system, coil3)

        # Test get_all_voltages (at t=0)
        all_voltages = get_all_voltages(system)
        @test all_voltages == [300.0, 400.0, 0.0]

        # Test get_powered_voltages
        powered_voltages = get_powered_voltages(system)
        @test powered_voltages == [300.0, 400.0]

        # Test get_controllable_voltages
        controllable_voltages = get_controllable_voltages(system)
        @test controllable_voltages == [300.0]

        # Test update_all_voltages!
        new_powered_voltages = [600.0, 700.0]
        update_all_voltages!(system, new_powered_voltages)
        @test get_powered_voltages(system) == new_powered_voltages

        # Test update_controllable_voltages!
        new_controllable_voltages = [800.0]
        update_controllable_voltages!(system, new_controllable_voltages)
        @test get_controllable_voltages(system) == new_controllable_voltages
        @test system.coils[2].voltage_ext == 700.0  # Non-controllable powered coil unchanged

        # Test error handling for wrong vector sizes
        @test_throws ErrorException update_all_voltages!(system, [1.0])       # Wrong size
        @test_throws ErrorException update_controllable_voltages!(system, [1.0, 2.0])  # Wrong size
    end

    @testset "Mixed System Operations" begin
        # Test a realistic mixed system
        system = CoilSystem{FT}()

        # Initialize with walls
        initialize_four_wall_system!(system, 20)

        # Add control coils with mixed controllability
        control_specs = [
            (r=2.5, z=0.4, area=π*0.02^2, name="PF1", max_voltage=1000.0, is_controllable=true),
            (r=1.8, z=0.0, area=π*0.03^2, name="CS", max_voltage=2000.0, is_controllable=true),
            (r=2.5, z=-0.4, area=π*0.02^2, name="PF2", max_voltage=1000.0, is_controllable=true),
            (r=1.2, z=0.0, area=π*0.015^2, name="heating", max_voltage=500.0, is_controllable=false)
        ]

        add_control_coils!(system, control_specs)

        @test system.n_total == 24  # 20 walls + 4 control coils
        @test system.n_powered == 4
        @test system.n_controllable == 3
        @test length(system.passive_indices) == 20

        # Test that controllable indices are correct
        controllable_coils = get_controllable_coils(system)
        controllable_names = [coil.name for coil in controllable_coils]
        @test "PF1" in controllable_names
        @test "CS" in controllable_names
        @test "PF2" in controllable_names
        @test "heating" ∉ controllable_names

        # Test vectorized operations work correctly
        controllable_currents = [5000.0, 20000.0, 4000.0]
        set_controllable_currents!(system, controllable_currents)
        @test get_controllable_currents(system) == controllable_currents

        # Heating coil current should be unchanged
        heating_idx = find_coil_by_name(system, "heating")
        @test system.coils[heating_idx].current == 0.0  # Default value
    end
end
