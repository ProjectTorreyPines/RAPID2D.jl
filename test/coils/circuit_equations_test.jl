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

        # Check that A_LR_circuit matrix has correct dimensions
        @test size(csys.A_LR_circuit) == (2, 2)
        @test size(csys.inv_A_LR_circuit) == (2, 2)

        # Check that A_LR_circuit = L + R*dt
        expected_A11 = csys.mutual_inductance[1,1] + csys.Δt * coil1.resistance
        expected_A22 = csys.mutual_inductance[2,2] + csys.Δt * coil2.resistance

        @test csys.A_LR_circuit[1,1] ≈ expected_A11
        @test csys.A_LR_circuit[2,2] ≈ expected_A22

        # Off-diagonal elements should be same as mutual inductance
        @test csys.A_LR_circuit[1,2] ≈ csys.mutual_inductance[1,2]
        @test csys.A_LR_circuit[2,1] ≈ csys.mutual_inductance[2,1]

        # Check that inv_A_LR_circuit is actually the inverse
        identity_check = csys.A_LR_circuit * csys.inv_A_LR_circuit
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
        @test size(csys.A_LR_circuit) == (2, 2)
        @test size(csys.inv_A_LR_circuit) == (2, 2)

        # Check that diagonal elements are correct
        @test csys.mutual_inductance[1,1] ≈ coil1.self_inductance
        @test csys.mutual_inductance[2,2] ≈ coil2.self_inductance

        # Check that A_LR_circuit = L + R*dt
        @test csys.A_LR_circuit[1,1] ≈ coil1.self_inductance + csys.Δt * coil1.resistance
        @test csys.A_LR_circuit[2,2] ≈ coil2.self_inductance + csys.Δt * coil2.resistance
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

@testset "Current Distribution Functions" begin
    FT = Float64
    @testset "Single Coil Current Distribution" begin
        # Create a simple 5x5 grid
        NR, NZ = 5, 5
        grid = GridGeometry{FT}(NR, NZ)

        # Initialize grid with R = [1, 2, 3, 4, 5], Z = [1, 2, 3, 4, 5]
        initialize_grid_geometry!(grid, (1.0, 5.0), (1.0, 5.0))

        # Create a single coil at grid point (2.5, 2.5) - exactly between grid points
        coil = Coil{FT}(
            position=(r=2.5, z=2.5),
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="test_coil"
        )

        coil_system = CoilSystem{FT}([coil])
        determine_coils_inside_grid!(coil_system, grid)

        # Set coil current to 1.0 A
        coil_system.coils[1].current = 1.0
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # Expected behavior: current should be distributed equally to 4 nodes
        # Grid spacing: dR = dZ = 1.0, so inv_dA = 1.0
        # Coil at (2.5, 2.5) should distribute to:
        # - [2,2]: (1-0.5)*(1-0.5) = 0.25
        # - [3,2]: 0.5*(1-0.5) = 0.25
        # - [2,3]: (1-0.5)*0.5 = 0.25
        # - [3,3]: 0.5*0.5 = 0.25
        expected_value = 0.25  # Current density = 1.0 A * 0.25 * inv_dA = 0.25 A/m²

        @test Jϕ[2,2] ≈ expected_value atol=1e-12
        @test Jϕ[3,2] ≈ expected_value atol=1e-12
        @test Jϕ[2,3] ≈ expected_value atol=1e-12
        @test Jϕ[3,3] ≈ expected_value atol=1e-12

        # All other points should be zero
        for i in 1:NR, j in 1:NZ
            if (i,j) ∉ [(2,2), (3,2), (2,3), (3,3)]
                @test Jϕ[i,j] ≈ 0.0 atol=1e-12
            end
        end

        # Test current conservation: sum of distributed current should equal input
        total_distributed = sum(Jϕ) * grid.dR * grid.dZ
        @test total_distributed ≈ 1.0 atol=1e-12
    end

    @testset "Coil at Grid Point" begin
        # Test coil exactly at a grid point
        NR, NZ = 4, 4
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 3.0), (0.0, 3.0))

        # Coil exactly at grid point (1.0, 1.0) - grid index [2,2]
        coil = Coil{FT}(
            position=(r=1.0, z=1.0),
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="grid_point_coil"
        )

        coil_system = CoilSystem{FT}([coil])
        determine_coils_inside_grid!(coil_system, grid)

        # Set coil current to 2.0 A
        coil_system.coils[1].current = 2.0
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # All current should go to the single grid point
        # dR = dZ = 1.0, so inv_dA = 1.0
        # Current density = 2.0 A * 1.0 * 1.0 = 2.0 A/m²
        @test Jϕ[2,2] ≈ 2.0 atol=1e-12

        # All other points should be zero
        for i in 1:NR, j in 1:NZ
            if (i,j) != (2,2)
                @test Jϕ[i,j] ≈ 0.0 atol=1e-12
            end
        end
    end

    @testset "Multiple Coils" begin
        # Test with multiple coils
        NR, NZ = 6, 6
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 5.0), (0.0, 5.0))

        # Create two coils at different positions
        coil1 = Coil{FT}(
            position=(r=1.5, z=1.5),  # Between grid points
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="coil1"
        )

        coil2 = Coil{FT}(
            position=(r=3.0, z=3.0),  # At grid point
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="coil2"
        )

        coil_system = CoilSystem{FT}([coil1, coil2])
        determine_coils_inside_grid!(coil_system, grid)

        # Set coil currents
        coil_system.coils[1].current = 1.0
        coil_system.coils[2].current = 2.0
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # Check that both coils contribute independently
        # Grid spacing: dR = dZ = 1.0

        # Coil1 at (1.5, 1.5) should contribute 0.25 to each of 4 surrounding nodes
        @test Jϕ[2,2] ≈ 0.25 atol=1e-12  # From coil1 only
        @test Jϕ[3,2] ≈ 0.25 atol=1e-12  # From coil1 only
        @test Jϕ[2,3] ≈ 0.25 atol=1e-12  # From coil1 only
        @test Jϕ[3,3] ≈ 0.25 atol=1e-12  # From coil1 only

        # Coil2 at (3.0, 3.0) should contribute 2.0 to node [4,4]
        @test Jϕ[4,4] ≈ 2.0 atol=1e-12   # From coil2 only

        # Test current conservation
        total_distributed = sum(Jϕ) * grid.dR * grid.dZ
        @test total_distributed ≈ 3.0 atol=1e-12  # 1.0 + 2.0
    end

    @testset "Boundary Conditions" begin
        # Test coils near/outside boundaries
        NR, NZ = 4, 4
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 3.0), (0.0, 3.0))

        # Create coils: inside, on boundary, and outside
        coils = [
            Coil{FT}(position=(r=1.5, z=1.5), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="inside"),
            Coil{FT}(position=(r=3.0, z=1.5), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="on_boundary"),
            Coil{FT}(position=(r=4.0, z=1.5), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="outside")
        ]

        coil_system = CoilSystem{FT}(coils)
        determine_coils_inside_grid!(coil_system, grid)

        # Only first two coils should be inside domain
        @test length(coil_system.inside_domain_indices) == 2
        @test 1 in coil_system.inside_domain_indices  # inside coil
        @test 2 in coil_system.inside_domain_indices  # on boundary coil
        @test 3 ∉ coil_system.inside_domain_indices   # outside coil

        # Set coil currents
        for (i, current) in enumerate([1.0, 1.0, 1.0])
            coil_system.coils[i].current = current
        end
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # Only inside and boundary coils should contribute
        total_distributed = sum(Jϕ) * grid.dR * grid.dZ
        @test total_distributed ≈ 2.0 atol=1e-12  # Only first two coils
    end

    @testset "Coil Mask Functionality" begin
        # Test selective coil processing using mask
        NR, NZ = 4, 4
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 3.0), (0.0, 3.0))

        # Create three coils inside domain
        coils = [
            Coil{FT}(position=(r=1.0, z=1.0), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="coil1"),
            Coil{FT}(position=(r=2.0, z=1.0), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="coil2"),
            Coil{FT}(position=(r=1.0, z=2.0), area=1.0, resistance=1.0, self_inductance=1.0, is_powered=true, name="coil3")
        ]

        coil_system = CoilSystem{FT}(coils)
        determine_coils_inside_grid!(coil_system, grid)

        # Set coil currents
        for (i, current) in enumerate([1.0, 1.0, 1.0])
            coil_system.coils[i].current = current
        end

        # Test with mask that includes only first and third coils
        mask = [true, false, true]
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid; coil_mask=mask)

        # Should only see contributions from coils 1 and 3
        total_distributed = sum(Jϕ) * grid.dR * grid.dZ
        @test total_distributed ≈ 2.0 atol=1e-12  # Only coils 1 and 3

        # Node [3,2] should be zero (where coil2 would contribute)
        @test Jϕ[3,2] ≈ 0.0 atol=1e-12
    end

    @testset "In-place vs Allocating Functions" begin
        # Test that both versions give same results
        NR, NZ = 4, 4
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 3.0), (0.0, 3.0))

        coil = Coil{FT}(
            position=(r=1.5, z=1.5),
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="test"
        )

        coil_system = CoilSystem{FT}([coil])
        determine_coils_inside_grid!(coil_system, grid)

        # Set coil current
        coil_system.coils[1].current = 1.0

        # Allocating version
        Jϕ_alloc = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # In-place version
        Jϕ_inplace = zeros(FT, NR, NZ)
        distribute_coil_currents_to_Jϕ!(Jϕ_inplace, coil_system, grid)

        # Should be identical
        @test Jϕ_alloc ≈ Jϕ_inplace atol=1e-12
    end

    @testset "Zero Current Handling" begin
        # Test handling of zero currents
        NR, NZ = 4, 4
        grid = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid, (0.0, 3.0), (0.0, 3.0))

        coil = Coil{FT}(
            position=(r=1.5, z=1.5),
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="zero_current"
        )

        coil_system = CoilSystem{FT}([coil])
        determine_coils_inside_grid!(coil_system, grid)

        # Set zero current
        coil_system.coils[1].current = 0.0
        Jϕ = distribute_coil_currents_to_Jϕ(coil_system, grid)

        # Should be all zeros
        @test all(Jϕ .≈ 0.0)
    end

    @testset "Grid Spacing Effects" begin
        # Test with different grid spacings
        NR, NZ = 4, 4

        # Fine grid
        grid_fine = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid_fine, (0.0, 1.0), (0.0, 1.0))  # dR = dZ = 1/3

        # Coarse grid
        grid_coarse = GridGeometry{FT}(NR, NZ)
        initialize_grid_geometry!(grid_coarse, (0.0, 3.0), (0.0, 3.0))  # dR = dZ = 1

        # Same coil position (fractionally)
        coil = Coil{FT}(
            position=(r=0.5, z=0.5),  # Relative position in fine grid
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="test"
        )

        coil_system_fine = CoilSystem{FT}([coil])
        determine_coils_inside_grid!(coil_system_fine, grid_fine)

        # Scale coil position for coarse grid
        coil_coarse = Coil{FT}(
            position=(r=1.5, z=1.5),  # Equivalent position in coarse grid
            area=1.0, resistance=1.0, self_inductance=1.0,
            is_powered=true, is_controllable=true,
            name="test"
        )

        coil_system_coarse = CoilSystem{FT}([coil_coarse])
        determine_coils_inside_grid!(coil_system_coarse, grid_coarse)

        # Set coil currents
        coil_system_fine.coils[1].current = 1.0
        coil_system_coarse.coils[1].current = 1.0

        Jϕ_fine = distribute_coil_currents_to_Jϕ(coil_system_fine, grid_fine)
        Jϕ_coarse = distribute_coil_currents_to_Jϕ(coil_system_coarse, grid_coarse)

        # Current conservation should hold for both
        total_fine = sum(Jϕ_fine) * grid_fine.dR * grid_fine.dZ
        total_coarse = sum(Jϕ_coarse) * grid_coarse.dR * grid_coarse.dZ

        @test total_fine ≈ 1.0 atol=1e-12
        @test total_coarse ≈ 1.0 atol=1e-12

        # Current density values should scale with inverse cell area
        # Fine grid has 9x smaller cells, so peak current density should be 9x higher
        @test maximum(Jϕ_fine) ≈ 9.0 * maximum(Jϕ_coarse) atol=1e-10
    end
end