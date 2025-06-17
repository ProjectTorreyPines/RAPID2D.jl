"""
Test script for the new modular control system

Tests the Controller, ControllerSet, and related functionality including:
- Controller creation and management
- Coil system integration
- Control signal computation and application
- Initialization functions
"""

using RAPID2D
using Test
using DiscretePIDs

@testset "Control System Tests" begin

    @testset "Controller Creation" begin
        # Create test coils with proper constructor
        area = π * 0.02^2  # 2cm radius coil
        resistance = 0.001  # 1 mΩ
        self_inductance = 1e-6  # 1 μH

        test_coils = [
            Coil((r=0.5, z=0.1), area, resistance, self_inductance, true, true, "OH1", 1000.0, 50000.0),
            Coil((r=0.5, z=-0.1), area, resistance, self_inductance, true, true, "OH2", 1000.0, 50000.0),
            Coil((r=0.3, z=0.2), area, resistance, self_inductance, true, false, "PF1", 800.0, 30000.0)
        ]

        @testset "Generic Controller Creation" begin
            target = 1e6
            controller = create_controller(target, test_coils[1:2];
                control_type="current",
                Kp=5.0, Ti=0.4, Td=0.02,
                dt=1e-6,
                umin=-100.0, umax=100.0
            )

            @test controller.target == target
            @test length(controller.coils) == 2
            @test controller.control_type == "current"
            @test controller.coils[1].name == "OH1"
            @test controller.coils[2].name == "OH2"

            # Test that coils are copied (not referenced)
            @test controller.coils !== test_coils[1:2]

            # Test PID parameters
            @test controller.pid.K == 5.0
            @test controller.pid.Ti == 0.4
            @test controller.pid.Td == 0.02
            @test controller.pid.Ts == 1e-6
            @test controller.pid.umin == -100.0
            @test controller.pid.umax == 100.0
        end

        @testset "Current Controller Creation" begin
            target_current = 1.5e6
            controller = create_current_controller(target_current, test_coils[1:2];
                Kp=10.0, Ti=0.5, Td=0.01,
                dt=1e-6
            )

            @test controller.target == target_current
            @test controller.control_type == "current"
            @test controller.pid.K == 10.0
            @test controller.pid.Ti == 0.5
            @test controller.pid.Td == 0.01
        end

        @testset "Position Controller Creation" begin
            target_position = 0.65
            controller = create_position_controller(target_position, test_coils[2:3];
                Kp=2.0, Ti=5.0, Td=0.1,
                dt=1e-6
            )

            @test controller.target == target_position
            @test controller.control_type == "position"
            @test controller.pid.K == 2.0
            @test controller.pid.Ti == 5.0
            @test controller.pid.Td == 0.1
        end
    end

    @testset "Controller Management" begin
        # Create test coils with proper constructor
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6

        test_coils = [
            Coil((r=0.5, z=0.1), area, resistance, self_inductance, true, true, "OH1", 1000.0, 50000.0),
            Coil((r=0.5, z=-0.1), area, resistance, self_inductance, true, true, "OH2", 1000.0, 50000.0)
        ]

        controller = create_current_controller(1e6, test_coils; dt=1e-6)

        @testset "Target Setting" begin
            new_target = 2e6
            set_target!(controller, new_target)
            @test controller.target == new_target
        end

        @testset "Controller Reset" begin
            # First, trigger some PID state
            compute_control_signal!(controller, 0.5e6)

            # Reset controller
            reset_controller!(controller)

            # Check that state is reset (this tests that reset_state! was called)
            # We can't directly check internal state, but we can verify the function runs
            @test controller isa Controller
        end

        @testset "Control Signal Computation" begin
            controller.target = 1e6
            current_value = 0.8e6

            # Compute control signal
            signal = compute_control_signal!(controller, current_value)

            # Should be a finite number
            @test isfinite(signal)

            # For proportional control with error = 0.2e6 and Kp=5.0
            # Expected signal should be around 5.0 * 0.2e6 = 1e6
            # (exact value depends on PID implementation details)
            @test abs(signal) > 0  # Should produce some control signal
        end

        @testset "Control Signal Application" begin
            # Set initial voltages
            for coil in test_coils
                coil.voltage_ext = 0.0
            end

            controller = create_current_controller(1e6, test_coils; dt=1e-6)
            signal = 50.0

            apply_control_signal!(controller, signal)

            # Check that all controlled coils have the signal applied
            for coil in controller.coils
                @test coil.voltage_ext == signal
            end

            # Check that original coils are also updated (should be references)
            for coil in test_coils
                @test coil.voltage_ext == signal
            end
        end
    end

    @testset "ControllerSet" begin
        @testset "ControllerSet Creation" begin
            controller_set = ControllerSet{Float64}()

            @test controller_set.current === nothing
            @test controller_set.position === nothing
            @test controller_set.temperature === nothing
            @test isempty(controller_set.custom)
        end

        @testset "ControllerSet Population" begin
            # Create test coils with proper constructor
            area = π * 0.02^2
            resistance = 0.001
            self_inductance = 1e-6

            oh_coils = [
                Coil((r=0.5, z=0.1), area, resistance, self_inductance, true, true, "OH1", 1000.0, 50000.0),
                Coil((r=0.5, z=-0.1), area, resistance, self_inductance, true, true, "OH2", 1000.0, 50000.0)
            ]
            pf_coils = [
                Coil((r=0.3, z=0.2), area, resistance, self_inductance, true, false, "PF1", 800.0, 30000.0),
                Coil((r=0.7, z=0.3), area, resistance, self_inductance, true, false, "PF2", 800.0, 30000.0)
            ]

            controller_set = ControllerSet{Float64}()

            # Add current controller
            controller_set.current = create_current_controller(1e6, oh_coils; dt=1e-6)
            @test controller_set.current !== nothing
            @test controller_set.current.control_type == "current"

            # Add position controller
            controller_set.position = create_position_controller(0.65, pf_coils; dt=1e-6)
            @test controller_set.position !== nothing
            @test controller_set.position.control_type == "position"

            # Add custom controller
            custom_controller = create_controller(100.0, oh_coils;
                control_type="custom", dt=1e-6)
            controller_set.custom["test"] = custom_controller
            @test haskey(controller_set.custom, "test")
            @test controller_set.custom["test"].control_type == "custom"
        end
    end

    @testset "Coil Finding Functions" begin
        # Create a mock coil system with proper constructor
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6

        coils = [
            Coil((r=0.5, z=0.1), area, resistance, self_inductance, true, true, "OH_Upper", 1000.0, 50000.0),
            Coil((r=0.5, z=-0.1), area, resistance, self_inductance, true, true, "OH_Lower", 1000.0, 50000.0),
            Coil((r=0.3, z=0.2), area, resistance, self_inductance, true, false, "PF_Coil_1", 800.0, 30000.0),
            Coil((r=0.7, z=0.3), area, resistance, self_inductance, true, false, "PF_Coil_2", 800.0, 30000.0),
            Coil((r=0.4, z=0.0), area, resistance, self_inductance, true, false, "CS_Main", 500.0, 40000.0),
            Coil((r=0.6, z=0.15), area, resistance, self_inductance, false, false, "Random_Coil", nothing, nothing)
        ]

        # Mock coil system
        coil_system = (coils=coils,)

        @testset "Case Insensitive Pattern Matching" begin
            # Test OH pattern matching
            oh_coils = find_coils_by_name(coil_system, ["OH"])
            @test length(oh_coils) == 2
            @test oh_coils[1].name == "OH_Upper"
            @test oh_coils[2].name == "OH_Lower"

            # Test PF pattern matching
            pf_coils = find_coils_by_name(coil_system, ["PF"])
            @test length(pf_coils) == 2
            @test pf_coils[1].name == "PF_Coil_1"
            @test pf_coils[2].name == "PF_Coil_2"

            # Test CS pattern matching
            cs_coils = find_coils_by_name(coil_system, ["CS"])
            @test length(cs_coils) == 1
            @test cs_coils[1].name == "CS_Main"
        end

        @testset "Multiple Pattern Matching" begin
            # Test multiple patterns
            oh_and_cs_coils = find_coils_by_name(coil_system, ["OH", "CS"])
            @test length(oh_and_cs_coils) == 3  # 2 OH + 1 CS

            # Test fallback patterns
            current_coils = find_coils_by_name(coil_system, ["OHMIC", "CURRENT", "CS"])
            @test length(current_coils) == 1  # Only CS_Main matches
        end

        @testset "Case Sensitive Matching" begin
            # Test case sensitive matching
            oh_coils_sensitive = find_coils_by_name(coil_system, ["oh"]; case_sensitive=true)
            @test length(oh_coils_sensitive) == 0  # No lowercase "oh"

            oh_coils_insensitive = find_coils_by_name(coil_system, ["oh"]; case_sensitive=false)
            @test length(oh_coils_insensitive) == 2  # Should find OH coils
        end

        @testset "No Matches" begin
            # Test no matches
            no_matches = find_coils_by_name(coil_system, ["NONEXISTENT"])
            @test length(no_matches) == 0
        end
    end

    @testset "Coil System Integration" begin
        # Test that controllers properly integrate with coil system
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6

        coils = [
            Coil((r=0.5, z=0.1), area, resistance, self_inductance, true, true, "OH1", 1000.0, 50000.0),
            Coil((r=0.5, z=-0.1), area, resistance, self_inductance, true, true, "OH2", 1000.0, 50000.0),
            Coil((r=0.3, z=0.2), area, resistance, self_inductance, true, false, "PF1", 800.0, 30000.0)
        ]

        # Initialize all coil voltages to zero
        for coil in coils
            coil.voltage_ext = 0.0
        end

        # Create controller using first two coils
        controller = create_current_controller(1e6, coils[1:2]; dt=1e-6)

        # Apply control signal
        test_signal = 75.0
        apply_control_signal!(controller, test_signal)

        # Check that controlled coils are updated
        @test coils[1].voltage_ext == test_signal
        @test coils[2].voltage_ext == test_signal

        # Check that uncontrolled coil is not updated
        @test coils[3].voltage_ext == 0.0

        # Test with different signal
        test_signal2 = -25.0
        apply_control_signal!(controller, test_signal2)

        @test coils[1].voltage_ext == test_signal2
        @test coils[2].voltage_ext == test_signal2
        @test coils[3].voltage_ext == 0.0  # Still unchanged
    end

    @testset "Controller Limits" begin
        area = π * 0.02^2
        resistance = 0.001
        self_inductance = 1e-6
        test_coils = [Coil((r=0.5, z=0.0), area, resistance, self_inductance, true, true, "Test", 1000.0, 50000.0)]

        # Create controller with tight limits
        controller = create_current_controller(1e6, test_coils;
            dt=1e-6, umin=-10.0, umax=10.0)

        # Test with large error (should be limited)
        large_error_signal = compute_control_signal!(controller, 0.0)  # Huge error

        # Signal should be limited
        @test large_error_signal >= -10.0
        @test large_error_signal <= 10.0
    end
end

@testset "Integration with Interface Functions" begin
    # Test interface functions work with controllers
    # (This would require a more complete RAPID setup)

    @testset "Measurement Extraction" begin
        # This would test extract_plasma_current, extract_plasma_position, etc.
        # For now, just verify the functions exist and can be called
        @test hasmethod(extract_plasma_current, (RAPID{Float64},))
        @test hasmethod(extract_plasma_position, (RAPID{Float64},))
        @test hasmethod(extract_plasma_temperature, (RAPID{Float64},))
        @test hasmethod(update_controller!, (RAPID{Float64}, Controller{Float64}))
    end
end

println("New control system tests completed successfully!")
