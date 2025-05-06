using Test
using RAPID2D

@testset "Utils Tests" begin
    @testset "Wall Status Functions" begin
        # Simple rectangular wall for testing
        # Coordinates: (1,1), (3,1), (3,3), (1,3), (1,1) - last matches first to close the loop
        wall_R = Float64[1.0, 3.0, 3.0, 1.0, 1.0]
        wall_Z = Float64[1.0, 1.0, 3.0, 3.0, 1.0]

        # Create WallGeometry object for the rectangle
        rect_wall = WallGeometry{Float64}(wall_R, wall_Z)

        # Complex tokamak-shaped wall (data from valid_wall.dat)
        test_dir = joinpath(@__DIR__, "..", "data", "wall")
        wall_file = joinpath(test_dir, "valid_wall.dat")

        # Load data from valid_wall.dat file
        tokamak_wall_data = RAPID2D.read_wall_data_file(wall_file)
        tokamak_wall_R = tokamak_wall_data.R
        tokamak_wall_Z = tokamak_wall_data.Z

        # Create WallGeometry object for the tokamak
        tokamak_wall = WallGeometry{Float64}(tokamak_wall_R, tokamak_wall_Z)

        @testset "get_wall_status - Basic Tests" begin
            # Test for points inside the wall
            @test RAPID2D.get_wall_status(2.0, 2.0, wall_R, wall_Z) == 1
            @test RAPID2D.get_wall_status(2.0, 2.0, rect_wall) == 1

            # Test for points outside the wall
            @test RAPID2D.get_wall_status(0.5, 2.0, rect_wall) == -1
            @test RAPID2D.get_wall_status(4.0, 2.0, rect_wall) == -1
            @test RAPID2D.get_wall_status(2.0, 0.5, rect_wall) == -1
            @test RAPID2D.get_wall_status(2.0, 4.0, rect_wall) == -1

            # Test for points on the boundary
            @test RAPID2D.get_wall_status(2.0, 1.0, rect_wall) == 0  # Bottom edge
            @test RAPID2D.get_wall_status(3.0, 2.0, rect_wall) == 0  # Right edge
            @test RAPID2D.get_wall_status(2.0, 3.0, rect_wall) == 0  # Top edge
            @test RAPID2D.get_wall_status(1.0, 2.0, rect_wall) == 0  # Left edge

            # Test for corner points
            @test RAPID2D.get_wall_status(1.0, 1.0, rect_wall) == 0
            @test RAPID2D.get_wall_status(3.0, 1.0, rect_wall) == 0
            @test RAPID2D.get_wall_status(3.0, 3.0, rect_wall) == 0
            @test RAPID2D.get_wall_status(1.0, 3.0, rect_wall) == 0
        end

        @testset "get_wall_status - Edge Cases" begin
            # Test with extreme values
            @test RAPID2D.get_wall_status(1e6, 1e6, wall_R, wall_Z) == -1  # Very large values
            @test RAPID2D.get_wall_status(1e6, 1e6, rect_wall) == -1  # Very large values
            @test RAPID2D.get_wall_status(-1e6, -1e6, wall_R, wall_Z) == -1  # Very small values
            @test RAPID2D.get_wall_status(-1e6, -1e6, rect_wall) == -1  # Very small values

            # Test with a very small polygon
            tiny_wall_R = Float64[0.0, 0.001, 0.001, 0.0, 0.0]
            tiny_wall_Z = Float64[0.0, 0.0, 0.001, 0.001, 0.0]
            tiny_wall = WallGeometry{Float64}(tiny_wall_R, tiny_wall_Z)

            @test RAPID2D.get_wall_status(0.0005, 0.0005, tiny_wall_R, tiny_wall_Z) == 1  # Inside
            @test RAPID2D.get_wall_status(0.0005, 0.0005, tiny_wall) == 1  # Inside
            @test RAPID2D.get_wall_status(0.002, 0.002, tiny_wall_R, tiny_wall_Z) == -1  # Outside
            @test RAPID2D.get_wall_status(0.002, 0.002, tiny_wall) == -1  # Outside
        end

        @testset "get_wall_status - Tokamak Wall" begin
            # Test with tokamak-shaped wall (using data from valid_wall.dat)
            # Check for point at the center
            center_R = sum(tokamak_wall_R) / length(tokamak_wall_R)
            center_Z = sum(tokamak_wall_Z) / length(tokamak_wall_Z)

            # The center point should be inside
            @test RAPID2D.get_wall_status(center_R, center_Z, tokamak_wall_R, tokamak_wall_Z) == 1
            @test RAPID2D.get_wall_status(center_R, center_Z, tokamak_wall) == 1

            # Check for point far outside the wall
            far_point_R = maximum(tokamak_wall_R) * 2
            far_point_Z = maximum(abs.(tokamak_wall_Z)) * 2
            @test RAPID2D.get_wall_status(far_point_R, far_point_Z, tokamak_wall_R, tokamak_wall_Z) == -1
            @test RAPID2D.get_wall_status(far_point_R, far_point_Z, tokamak_wall) == -1
        end

        @testset "is_inside_wall - Single Point" begin
            # Test for points inside the wall
            @test RAPID2D.is_inside_wall(2.0, 2.0, wall_R, wall_Z) == true
            @test RAPID2D.is_inside_wall(2.0, 2.0, rect_wall) == true

            # Test for points outside the wall
            @test RAPID2D.is_inside_wall(0.5, 2.0, wall_R, wall_Z) == false
            @test RAPID2D.is_inside_wall(0.5, 2.0, rect_wall) == false

            # Test for points on the boundary (boundary is considered inside)
            @test RAPID2D.is_inside_wall(2.0, 1.0, wall_R, wall_Z) == true
            @test RAPID2D.is_inside_wall(2.0, 1.0, rect_wall) == true
            @test RAPID2D.is_inside_wall(1.0, 1.0, wall_R, wall_Z) == true
            @test RAPID2D.is_inside_wall(1.0, 1.0, rect_wall) == true
        end

        @testset "is_inside_wall - Array Version (Vector)" begin
            # Test with vector input
            test_R = Float64[0.5, 1.5, 2.0, 2.5, 3.5]
            test_Z = Float64[2.0, 2.0, 2.0, 2.0, 2.0]

            expected = Bool[false, true, true, true, false]
            result1 = RAPID2D.is_inside_wall(test_R, test_Z, wall_R, wall_Z)
            result2 = RAPID2D.is_inside_wall(test_R, test_Z, rect_wall)

            @test result1 == expected
            @test result2 == expected
            @test size(result1) == size(test_R)
            @test size(result2) == size(test_R)
        end

        @testset "is_inside_wall - Array Version (Matrix)" begin
            # Test with matrix input
            test_R = Float64[
                1.5 2.0 2.5;
                1.5 2.0 2.5;
                1.5 2.0 2.5
            ]

            test_Z = Float64[
                0.5 0.5 0.5;
                2.0 2.0 2.0;
                3.5 3.5 3.5
            ]

            expected = Bool[
                false false false;
                true true true;
                false false false
            ]

            result1 = RAPID2D.is_inside_wall(test_R, test_Z, wall_R, wall_Z)
            result2 = RAPID2D.is_inside_wall(test_R, test_Z, rect_wall)

            @test result1 == expected
            @test result2 == expected
            @test size(result1) == size(test_R)
            @test size(result2) == size(test_R)
        end

        @testset "is_inside_wall - Error Handling" begin
            # Test with R, Z arrays of different sizes
            test_R1 = Float64[1.0, 2.0, 3.0]
            test_Z1 = Float64[1.0, 2.0]

            @test_throws ArgumentError RAPID2D.is_inside_wall(test_R1, test_Z1, wall_R, wall_Z)
            @test_throws ArgumentError RAPID2D.is_inside_wall(test_R1, test_Z1, rect_wall)
        end
    end
end