using Test
using RAPID2D

@testset "Circuit Equations Tests" begin
    FT = Float64

    @testset "Basic Mutual Inductance Calculation" begin
        # Create a simple coil csys with 3 coils
        csys = CoilSystem{FT}()

        # Add test coils in a simple configuration
        coil1 = Coil((r=2.0, z=0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        coil2 = Coil((r=2.0, z=0.0), π*0.02^2, 0.001, 1e-6, true, true, "CS", 2000.0, 100000.0)
        coil3 = Coil((r=2.0, z=-0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF2", 1000.0, 50000.0)

        add_coil!(csys, coil1)
        add_coil!(csys, coil2)
        add_coil!(csys, coil3)

        @test csys.n_total == 3

        # Calculate mutual inductance matrix
        calculate_mutual_inductance_matrix!(csys)

        # Check that mutual inductance matrix has correct dimensions
        @test size(csys.mutual_inductance) == (3, 3)

        # Check that diagonal elements are self-inductances
        @test csys.mutual_inductance[1,1] ≈ coil1.self_inductance
        @test csys.mutual_inductance[2,2] ≈ coil2.self_inductance
        @test csys.mutual_inductance[3,3] ≈ coil3.self_inductance

        # Check symmetry of mutual inductance matrix
        @test csys.mutual_inductance[1,2] ≈ csys.mutual_inductance[2,1]
        @test csys.mutual_inductance[1,3] ≈ csys.mutual_inductance[3,1]
        @test csys.mutual_inductance[2,3] ≈ csys.mutual_inductance[3,2]

        # Check that off-diagonal elements are non-zero (mutual coupling exists)
        @test abs(csys.mutual_inductance[1,2]) > 0
        @test abs(csys.mutual_inductance[1,3]) > 0
        @test abs(csys.mutual_inductance[2,3]) > 0
    end

    @testset "Circuit Matrix Calculation" begin
        csys = CoilSystem{FT}()

        # Add coils with different resistances
        coil1 = Coil((r=2.0, z=0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        coil2 = Coil((r=1.8, z=0.0), π*0.015^2, 0.002, 2e-6, true, true, "CS", 2000.0, 100000.0)

        add_coil!(csys, coil1)
        add_coil!(csys, coil2)

        # First calculate mutual inductance
        calculate_mutual_inductance_matrix!(csys)

        # Then calculate circuit matrices with a time step
        csys.Δt = 1e-5  # 10 microseconds
        calculate_circuit_matrices!(csys)

        # Check that A_circuit matrix has correct dimensions
        @test size(csys.A_circuit) == (2, 2)
        @test size(csys.inv_A_circuit) == (2, 2)

        # Check that A_circuit = L + R*dt
        expected_A11 = csys.mutual_inductance[1,1] + csys.Δt * coil1.resistance
        expected_A22 = csys.mutual_inductance[2,2] + csys.Δt * coil2.resistance

        @test csys.A_circuit[1,1] ≈ expected_A11
        @test csys.A_circuit[2,2] ≈ expected_A22

        # Off-diagonal elements should be same as mutual inductance
        @test csys.A_circuit[1,2] ≈ csys.mutual_inductance[1,2]
        @test csys.A_circuit[2,1] ≈ csys.mutual_inductance[2,1]

        # Check that inv_A_circuit is actually the inverse
        identity_check = csys.A_circuit * csys.inv_A_circuit
        @test identity_check ≈ RAPID2D.LinearAlgebra.I(2) atol=1e-12
    end

    @testset "Update System Matrices" begin
        csys = CoilSystem{FT}()

        # Add a few coils
        coil1 = Coil((r=2.0, z=0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        coil2 = Coil((r=2.0, z=-0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF2", 1000.0, 50000.0)

        add_coil!(csys, coil1)
        add_coil!(csys, coil2)

        csys.Δt = 1e-5

        # Test the convenience function that updates everything
        update_coil_system_matrices!(csys)

        # Should have computed both mutual inductance and circuit matrices
        @test size(csys.mutual_inductance) == (2, 2)
        @test size(csys.A_circuit) == (2, 2)
        @test size(csys.inv_A_circuit) == (2, 2)

        # Check that diagonal elements are correct
        @test csys.mutual_inductance[1,1] ≈ coil1.self_inductance
        @test csys.mutual_inductance[2,2] ≈ coil2.self_inductance

        # Check that A_circuit = L + R*dt
        @test csys.A_circuit[1,1] ≈ coil1.self_inductance + csys.Δt * coil1.resistance
        @test csys.A_circuit[2,2] ≈ coil2.self_inductance + csys.Δt * coil2.resistance
    end

    @testset "Mutual Inductance Access Functions" begin
        csys = CoilSystem{FT}()

        coil1 = Coil((r=2.0, z=0.5), π*0.02^2, 0.001, 1e-6, true, true, "PF1", 1000.0, 50000.0)
        coil2 = Coil((r=1.8, z=0.0), π*0.02^2, 0.001, 2e-6, true, true, "CS", 2000.0, 100000.0)

        add_coil!(csys, coil1)
        add_coil!(csys, coil2)

        calculate_mutual_inductance_matrix!(csys)

        # Test get_mutual_inductance function
        M12 = get_mutual_inductance(csys, 1, 2)
        @test M12 ≈ csys.mutual_inductance[1, 2]

        M21 = get_mutual_inductance(csys, 2, 1)
        @test M21 ≈ csys.mutual_inductance[2, 1]
        @test M12 ≈ M21  # Symmetry

        # Test get_coil_coupling_matrix function
        coupling_matrix = get_coil_coupling_matrix(csys)
        @test coupling_matrix == csys.mutual_inductance
        @test coupling_matrix !== csys.mutual_inductance  # Should be a copy

        # Test bounds checking
        @test_throws AssertionError get_mutual_inductance(csys, 3, 1)
        @test_throws AssertionError get_mutual_inductance(csys, 1, 3)
    end

    @testset "Empty System Handling" begin
        # Test that empty systems are handled gracefully
        csys = CoilSystem{FT}()

        # Should not crash with empty csys
        calculate_mutual_inductance_matrix!(csys)
        calculate_circuit_matrices!(csys)
        update_coil_system_matrices!(csys)

        # Matrices should remain empty/uninitialized for empty csys
        @test csys.n_total == 0
    end
end
