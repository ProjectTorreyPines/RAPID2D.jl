using Test
using RAPID2D

@testset "Time-dependent Voltage Functions" begin
    FT = Float64

    @testset "Voltage Function Evaluation" begin
        # Test constant voltage evaluation
        @test evaluate_voltage_ext(500.0, 1.0) == 500.0
        @test evaluate_voltage_ext(0.0, 5.0) == 0.0

        # Test function voltage evaluation
        linear_func = t -> 100.0 * t + 50.0
        @test evaluate_voltage_ext(linear_func, 0.0) == 50.0
        @test evaluate_voltage_ext(linear_func, 2.0) == 250.0

        sine_func = t -> 1000.0 * sin(2π * t)
        @test evaluate_voltage_ext(sine_func, 0.0) ≈ 0.0 atol=1e-10
        @test evaluate_voltage_ext(sine_func, 0.25) ≈ 1000.0 atol=1e-10
    end

    @testset "Individual Coil Time-dependent Voltage" begin
        coil_constant = Coil((r=2.0, z=0.0), 0.01, 0.001, 1e-6, true, true, "constant",
                            1000.0, 50000.0, 0.0, 750.0)

        # Test constant voltage at different times
        @test get_coil_voltage_at_time(coil_constant, 0.0) == 750.0
        @test get_coil_voltage_at_time(coil_constant, 1.0) == 750.0
        @test get_coil_voltage_at_time(coil_constant, 5.0) == 750.0

        # Test function voltage
        ramp_func = t -> 200.0 * t + 100.0
        coil_function = Coil((r=2.0, z=0.0), 0.01, 0.001, 1e-6, true, true, "function",
                            1000.0, 50000.0, 0.0, ramp_func)

        @test get_coil_voltage_at_time(coil_function, 0.0) == 100.0
        @test get_coil_voltage_at_time(coil_function, 1.0) == 300.0
        @test get_coil_voltage_at_time(coil_function, 2.5) == 600.0

        # Test passive coil (should always return 0)
        passive_coil = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "passive",
                           nothing, nothing, 0.0, 0.0)
        @test get_coil_voltage_at_time(passive_coil, 0.0) == 0.0
        @test get_coil_voltage_at_time(passive_coil, 10.0) == 0.0
    end

    @testset "System-level Time-dependent Voltage" begin
        system = CoilSystem{FT}()

        # Add coils with different voltage types
        coil1 = Coil((r=2.5, z=0.5), 0.01, 0.001, 1e-6, true, true, "constant_coil",
                    1000.0, 50000.0, 0.0, 400.0)

        ramp_func = t -> 100.0 * t
        coil2 = Coil((r=2.5, z=-0.5), 0.01, 0.001, 1e-6, true, false, "ramp_coil",
                    1000.0, 50000.0, 0.0, ramp_func)

        passive_coil = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall",
                           nothing, nothing, 0.0, 0.0)

        add_coil!(system, coil1)
        add_coil!(system, coil2)
        add_coil!(system, passive_coil)

        # Test get_coil_voltage_at_time by name
        @test get_coil_voltage_at_time(system, "constant_coil", 2.0) == 400.0
        @test get_coil_voltage_at_time(system, "ramp_coil", 2.0) == 200.0
        @test get_coil_voltage_at_time(system, "wall", 2.0) == 0.0

        # Test get_all_voltages_at_time
        all_voltages_t0 = get_all_voltages_at_time(system, 0.0)
        @test all_voltages_t0 == [400.0, 0.0, 0.0]

        all_voltages_t3 = get_all_voltages_at_time(system, 3.0)
        @test all_voltages_t3 == [400.0, 300.0, 0.0]

        # Test get_powered_voltages_at_time
        powered_voltages_t1 = get_powered_voltages_at_time(system, 1.0)
        @test powered_voltages_t1 == [400.0, 100.0]

        # Test get_controllable_voltages_at_time
        controllable_voltages_t2 = get_controllable_voltages_at_time(system, 2.0)
        @test controllable_voltages_t2 == [400.0]

        # Test backward compatibility (t=0 default)
        @test get_all_voltages(system) == get_all_voltages_at_time(system, 0.0)
        @test get_powered_voltages(system) == get_powered_voltages_at_time(system, 0.0)
        @test get_controllable_voltages(system) == get_controllable_voltages_at_time(system, 0.0)
    end

    @testset "Setting Voltage Functions" begin
        system = CoilSystem{FT}()

        coil = Coil((r=2.5, z=0.0), 0.01, 0.001, 1e-6, true, true, "test_coil",
                   1000.0, 50000.0, 0.0, 200.0)
        passive_coil = Coil((r=2.01, z=0.0), 0.005, 0.01, 1e-7, false, false, "wall",
                           nothing, nothing, 0.0, 0.0)

        add_coil!(system, coil)
        add_coil!(system, passive_coil)

        # Test setting voltage function
        sine_func = t -> 500.0 * sin(2π * 0.1 * t)  # 0.1 Hz
        set_coil_voltage_function!(system, "test_coil", sine_func)

        # Verify function was set
        @test get_coil_voltage_at_time(system, "test_coil", 0.0) ≈ 0.0 atol=1e-10
        @test get_coil_voltage_at_time(system, "test_coil", 2.5) ≈ 500.0 atol=1e-10
        @test get_coil_voltage_at_time(system, "test_coil", 5.0) ≈ 0.0 atol=1e-10

        # Test error handling
        @test_throws ErrorException set_coil_voltage_function!(system, "NonExistent", sine_func)
        @test_throws ErrorException set_coil_voltage_function!(system, "wall", sine_func)
    end

    @testset "Voltage Function Factories" begin
        # Test create_linear_voltage_ramp
        ramp = create_linear_voltage_ramp(150.0, 0.5, 100.0)
        @test ramp(0.0) == 100.0 - 150.0 * 0.5  # V0 + rate * (t - t_start)
        @test ramp(0.5) == 100.0
        @test ramp(1.5) == 100.0 + 150.0 * 1.0

        # Test create_sinusoidal_voltage
        sine = create_sinusoidal_voltage(800.0, 2.0, π/4, 50.0)  # 2 Hz, π/4 phase, 50V offset
        @test sine(0.0) ≈ 800.0 * sin(π/4) + 50.0
        @test sine(0.125) ≈ 800.0 * sin(π/2 + π/4) + 50.0  # quarter period later

        # Test create_step_voltage
        step = create_step_voltage(200.0, 800.0, 1.0)
        @test step(0.5) == 200.0
        @test step(1.0) == 800.0
        @test step(1.5) == 800.0

        # Test create_piecewise_linear_voltage
        times = [0.0, 1.0, 3.0, 4.0]
        voltages = [0.0, 500.0, 500.0, -200.0]
        piecewise = create_piecewise_linear_voltage(times, voltages)

        @test piecewise(-1.0) == 0.0     # Before range
        @test piecewise(0.0) == 0.0      # At first point
        @test piecewise(0.5) == 250.0    # Linear interpolation
        @test piecewise(1.0) == 500.0    # At point
        @test piecewise(2.0) == 500.0    # Constant segment
        @test piecewise(3.5) == 150.0    # Linear interpolation
        @test piecewise(5.0) == -200.0   # After range
    end

    @testset "Warning for Function Voltage without Time" begin
        system = CoilSystem{FT}()

        func_voltage = t -> 100.0 * t
        coil = Coil((r=2.5, z=0.0), 0.01, 0.001, 1e-6, true, true, "func_coil",
                   1000.0, 50000.0, 0.0, func_voltage)

        add_coil!(system, coil)

        # This should issue a warning and return the function itself
        result = get_coil_voltage(system, "func_coil")
        @test result isa Function
    end
end
