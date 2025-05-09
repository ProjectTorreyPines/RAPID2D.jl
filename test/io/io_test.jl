using Test
using RAPID2D

@testset "IO Tests" begin
    @testset "Wall Data Reading Tests" begin
        test_dir = joinpath(@__DIR__, "..", "data", "wall")

        @testset "read_wall_data_file - Valid Data" begin
            # Test reading a valid wall data file
            wall_file = joinpath(test_dir, "valid_wall.dat")
            wall = RAPID2D.read_wall_data_file(wall_file)

            # Check that the results are correct
            @test length(wall.R) == 13  # 12 points + 1 for closing the loop
            @test length(wall.Z) == 13
            # Check some specific points from the tokamak shape
            @test wall.R[1] ≈ 1.5
            @test wall.Z[1] ≈ 0.5
            @test wall.R[6] ≈ 0.7
            @test wall.Z[6] ≈ 0.0
            @test wall.R[12] ≈ 1.7
            @test wall.Z[12] ≈ 0.0
            # Check that the last point equals the first (closed loop)
            @test wall.R[end] == wall.R[1]
            @test wall.Z[end] == wall.Z[1]
        end

        @testset "read_wall_data_file - Fewer Points" begin
            # Test reading a file with fewer points than declared
            wall_file = joinpath(test_dir, "missing_wall.dat")

            # Define wall variable outside the @test_logs block so it's accessible later
            local wall

            # Should produce a warning but still read available data
            # We capture warnings using the @test_logs macro from Test
            @test_logs (:warn, r"Expected 10 points but only read 6 points from wall data file") begin
                wall = RAPID2D.read_wall_data_file(wall_file)
            end

            # Check that we got the correct data that was available
            @test length(wall.R) == 7  # 6 points + 1 for closing the loop
            @test wall.R[1] ≈ 1.5
            @test wall.Z[6] ≈ 0.0
            @test wall.R[end] == wall.R[1]  # Closed loop
        end

        @testset "read_wall_data_file - Error Handling" begin
            # Test reading a corrupted file with invalid header
            corrupt_file = joinpath(test_dir, "corrupt_wall.dat")

            # Should throw an error about invalid format
            @test_throws ErrorException RAPID2D.read_wall_data_file(corrupt_file)

            # Test with a non-existent file
            missing_file = joinpath(test_dir, "nonexistent.dat")
            @test_throws AssertionError RAPID2D.read_wall_data_file(missing_file)
        end

        @testset "read_device_wall_data" begin
            # Create a minimal RAPID instance for testing
            FT = Float64
            RP = RAPID2D.RAPID{FT}(10, 10)
            RP.config.Input_path = test_dir
            RP.config.device_Name = "valid"  # Will try to read valid_First_Wall.dat

            # Test with an explicit file
            wall_file = joinpath(test_dir, "valid_wall.dat")
            RAPID2D.read_device_wall_data!(RP, wall_file)

            # Check that the wall was loaded correctly
            @test length(RP.wall.R) == 13  # 12 points + 1 for closing the loop
            @test RP.wall.R[1] ≈ 1.5
            @test RP.wall.Z[1] ≈ 0.5
        end
    end
end