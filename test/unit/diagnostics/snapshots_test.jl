using RAPID2D
using Test


@testset "Snapshot Diagnostics Tests" begin

    @testset "Memory Safety: Reference vs Copy" begin
        # This is the most critical test - ensures data integrity
        dims_RZ = (20, 25)

        # Create mock data
        mock_ne = rand(dims_RZ...) * 1e19
        mock_Te = rand(dims_RZ...) * 8.0

        # Test Snapshot2D memory isolation
        snap2D_1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
        snap2D_2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

        # Use .= for safe copying (what we want)
        snap2D_1.ne .= mock_ne
        snap2D_2.ne .= mock_ne .* 2.0

        # Modify original data
        mock_ne[1,1] = 9.99e20

        # Snapshots should be unaffected (proper memory isolation)
        @test snap2D_1.ne[1,1] != 9.99e20
        @test snap2D_2.ne[1,1] != 9.99e20
        @test snap2D_1.ne !== snap2D_2.ne  # Different memory locations

        # Test that each snapshot maintains independent data
        snap2D_1.ne[2,2] = 1.23e19
        @test snap2D_2.ne[2,2] != 1.23e19  # Should not be affected
    end

    @testset "Snapshot0D Independence" begin
        # Test scalar field independence
        snap1 = Snapshot0D{Float64}()
        snap2 = Snapshot0D{Float64}()

        snap1.ne = 1.5e19
        snap1.Te_eV = 5.5
        snap1.step = 100

        snap2.ne = 2.8e19
        snap2.Te_eV = 8.2
        snap2.step = 200

        # Verify independence
        @test snap1.ne ≈ 1.5e19
        @test snap2.ne ≈ 2.8e19
        @test snap1.step == 100
        @test snap2.step == 200

        # Modify one, other should be unaffected
        snap1.ne = 9.99e19
        @test snap2.ne ≈ 2.8e19
    end

    @testset "Diagnostics Vector Operations" begin
        dims_RZ = (15, 20)
        diag = Diagnostics{Float64}(dims_RZ[1], dims_RZ[2])

        # Test initial state
        @test length(diag.snaps0D) == 0
        @test length(diag.snaps2D) == 0
        @test diag.tid_0D == 0
        @test diag.tid_2D == 0

        # Test dynamic growth
        for i in 1:5
            snap0D = Snapshot0D{Float64}()
            snap0D.step = i * 100
            snap0D.ne = i * 1e18
            push!(diag.snaps0D, snap0D)
            diag.tid_0D += 1
        end

        @test length(diag.snaps0D) == 5
        @test diag.tid_0D == 5
        @test diag.snaps0D[3].step == 300
        @test diag.snaps0D[3].ne ≈ 3e18

        # Test that snapshots in vector are independent
        diag.snaps0D[1].ne = 9.99e18
        @test diag.snaps0D[2].ne ≈ 2e18  # Should not be affected
    end

    @testset "Type Consistency" begin
        dims_RZ = (10, 15)

        # Test Float64
        diag_f64 = Diagnostics{Float64}(dims_RZ[1], dims_RZ[2])
        snap0D_f64 = Snapshot0D{Float64}()
        snap2D_f64 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

        @test typeof(snap0D_f64.ne) == Float64
        @test eltype(snap2D_f64.ne) == Float64

        # Test Float32
        diag_f32 = Diagnostics{Float32}(dims_RZ[1], dims_RZ[2])
        snap0D_f32 = Snapshot0D{Float32}()
        snap2D_f32 = Snapshot2D{Float32}(dims_RZ = dims_RZ)

        @test typeof(snap0D_f32.ne) == Float32
        @test eltype(snap2D_f32.ne) == Float32
    end

    @testset "Edge Cases and Error Handling" begin
        dims_RZ = (5, 8)

        # Test empty diagnostics
        diag = Diagnostics{Float64}(dims_RZ[1], dims_RZ[2])
        @test isempty(diag.snaps0D)
        @test isempty(diag.snaps2D)

        # Test SrcLossTracker initialization
        @test diag.Ntracker.dims_RZ == dims_RZ
        @test diag.Ntracker.cum0D_Ne_src == 0.0
        @test size(diag.Ntracker.cum2D_Ne_src) == dims_RZ

        # Test Snapshot2D with correct dimensions
        snap2D = Snapshot2D{Float64}(dims_RZ = dims_RZ)
        @test size(snap2D.ne) == dims_RZ
        @test size(snap2D.Te_eV) == dims_RZ
        @test size(snap2D.BR) == dims_RZ

        # Test that matrices are zero-initialized
        @test all(snap2D.ne .== 0.0)
        @test all(snap2D.Te_eV .== 0.0)
    end

    @testset "Pre-allocation Constructor" begin
        dims_RZ = (12, 18)
        n_0D, n_2D = 50, 10

        diag = Diagnostics{Float64}(dims_RZ[1], dims_RZ[2], n_0D, n_2D)

        # Verify pre-allocation
        @test length(diag.snaps0D) == n_0D
        @test length(diag.snaps2D) == n_2D
        @test diag.tid_0D == 0  # No snapshots recorded yet
        @test diag.tid_2D == 0

        # Verify that pre-allocated snapshots are properly initialized
        @test all(snap -> snap.step == 0, diag.snaps0D)
        @test all(snap -> snap.time_s == 0.0, diag.snaps0D)
        @test all(snap -> snap.dims_RZ == dims_RZ, diag.snaps2D)
        @test all(snap -> size(snap.ne) == dims_RZ, diag.snaps2D)

        # Test that we can modify pre-allocated snapshots independently
        diag.snaps0D[1].ne = 1.5e19
        diag.snaps0D[2].ne = 2.8e19
        @test diag.snaps0D[1].ne ≈ 1.5e19
        @test diag.snaps0D[2].ne ≈ 2.8e19
    end

    @testset "Critical Broadcasting Safety" begin
        # Test the .= vs = distinction that caused our memory issues
        dims_RZ = (8, 12)

        snap2D = Snapshot2D{Float64}(dims_RZ = dims_RZ)
        test_data = rand(dims_RZ...) * 1e19

        # Safe assignment (creates copy)
        snap2D.ne .= test_data
        original_value = test_data[1,1]

        # Modify original data
        test_data[1,1] = 9.99e20

        # Snapshot should retain original value (not affected by reference)
        @test snap2D.ne[1,1] ≈ original_value
        @test snap2D.ne[1,1] != 9.99e20

        # Test that snap2D.ne is actually a copy, not a reference
        snap2D.ne[2,2] = 7.77e19
        @test test_data[2,2] != 7.77e19  # Original data should be unaffected
    end
end


@testset "Snapshot Equality Operators" begin

    @testset "Snapshot0D Equality Tests" begin
        # Basic equality tests
        @testset "Basic Equality" begin
            snap1 = Snapshot0D{Float64}()
            snap2 = Snapshot0D{Float64}()

            # Test basic equality on empty snapshots
            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)
        end

        @testset "Scalar Field Equality" begin
            snap1 = Snapshot0D{Float64}()
            snap2 = Snapshot0D{Float64}()

            # Set identical values
            snap1.ne = 1.5e19
            snap1.Te_eV = 5.0
            snap1.step = 100
            snap1.time_s = 1.23

            snap2.ne = 1.5e19
            snap2.Te_eV = 5.0
            snap2.step = 100
            snap2.time_s = 1.23

            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test inequality
            snap2.ne = 2.0e19
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)
            @test !isapprox(snap1, snap2)
        end

        @testset "NaN Handling" begin
            snap1 = Snapshot0D{Float64}()
            snap2 = Snapshot0D{Float64}()

            # Set other fields to identical values
            snap1.Te_eV = 5.0
            snap1.step = 100
            snap2.Te_eV = 5.0
            snap2.step = 100

            # Test NaN handling
            snap1.ne = NaN
            snap2.ne = NaN

            # With ==, NaN != NaN (standard Julia behavior)
            @test !(snap1 == snap2)

            # With isequal, NaN == NaN (this is what we want for testing!)
            @test isequal(snap1, snap2)

            # With isapprox, depends on nans parameter
            @test !isapprox(snap1, snap2)  # nans=false by default
            @test isapprox(snap1, snap2, nans=true)  # nans=true treats NaN as equal
        end

        @testset "Dictionary Field Equality" begin
            snap1 = Snapshot0D{Float64}()
            snap2 = Snapshot0D{Float64}()

            # Test with empty CFL dictionaries (default)
            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test with populated dictionaries
            snap1.CFL = Dict(:electron => 0.5, :ion => 0.3)
            snap2.CFL = Dict(:electron => 0.5, :ion => 0.3)

            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test dictionary inequality
            snap2.CFL = Dict(:electron => 0.6, :ion => 0.3)
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)
            @test !isapprox(snap1, snap2)
        end

        @testset "Optional Array Field Equality" begin
            snap1 = Snapshot0D{Float64}()
            snap2 = Snapshot0D{Float64}()

            # Test with nothing arrays (default)
            @test snap1.coils_I === nothing
            @test snap2.coils_I === nothing
            @test snap1 == snap2
            @test isequal(snap1, snap2)

            # Test with populated arrays
            snap1.coils_I = [1.0, 2.0, 3.0, 4.0]
            snap2.coils_I = [1.0, 2.0, 3.0, 4.0]

            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test one nothing, one array
            snap2.coils_I = nothing
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)
            @test !isapprox(snap1, snap2)
        end

        @testset "Type Consistency" begin
            snap_f64 = Snapshot0D{Float64}()
            snap_f32 = Snapshot0D{Float32}()

            # Different float types should still be comparable if values are the same
            snap_f64.ne = 1.5e19
            snap_f32.ne = 1.5f19

            # Note: These may not be exactly equal due to precision differences
            # but the operators should work without throwing errors
            @test !(snap_f64 == snap_f32)
            @test !isequal(snap_f64, snap_f32)
            @test isapprox(snap_f64, snap_f32)
        end
    end

    @testset "Snapshot2D Equality Tests" begin
        @testset "Basic Equality" begin
            dims_RZ = (10, 15)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Test basic equality on zero-initialized snapshots
            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)
        end

        @testset "Dimension Mismatch" begin
            dims1 = (10, 15)
            dims2 = (12, 15)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims1)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims2)

            # Different dimensions should not be equal
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)
            @test !isapprox(snap1, snap2)
        end

        @testset "Matrix Field Equality" begin
            dims_RZ = (8, 12)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Set some matrix values
            snap1.ne[1,1] = 1.5e19
            snap1.Te_eV[2,3] = 8.0
            snap1.step = 100
            snap1.time_s = 1.23

            snap2.ne[1,1] = 1.5e19
            snap2.Te_eV[2,3] = 8.0
            snap2.step = 100
            snap2.time_s = 1.23

            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test matrix inequality
            snap2.ne[1,1] = 2.0e19
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)
            @test !isapprox(snap1, snap2)
        end

        @testset "NaN Handling in Matrices" begin
            dims_RZ = (6, 8)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Set some regular values
            snap1.ne[1,1] = 1.5e19
            snap2.ne[1,1] = 1.5e19

            # Test NaN handling in matrices
            snap1.ne[3,3] = NaN
            snap2.ne[3,3] = NaN

            # With ==, NaN != NaN
            @test !(snap1 == snap2)

            # With isequal, NaN == NaN
            @test isequal(snap1, snap2)

            # With isapprox
            @test !isapprox(snap1, snap2)  # nans=false by default
            @test isapprox(snap1, snap2, nans=true)  # nans=true treats NaN as equal
        end

        @testset "Approximate Equality" begin
            dims_RZ = (5, 7)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Set slightly different values
            snap1.ne[1,1] = 1.0000001e19
            snap2.ne[1,1] = 1.0000002e19

            snap1.Te_eV[2,2] = 5.0000001
            snap2.Te_eV[2,2] = 5.0000002

            # Should not be exactly equal
            @test !(snap1 == snap2)
            @test !isequal(snap1, snap2)

            # But should be approximately equal
            @test isapprox(snap1, snap2, rtol=1e-6)
            @test !isapprox(snap1, snap2, rtol=1e-8)
        end

        @testset "Scalar vs Matrix Field Types" begin
            dims_RZ = (4, 6)
            snap1 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap2 = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Test scalar fields
            snap1.step = 100
            snap1.time_s = 1.23
            snap2.step = 100
            snap2.time_s = 1.23

            @test snap1 == snap2
            @test isequal(snap1, snap2)
            @test isapprox(snap1, snap2)

            # Test mixed changes
            snap1.step = 200  # Change scalar
            @test !(snap1 == snap2)

            snap1.step = 100  # Reset
            snap1.ne[1,1] = 1e19  # Change matrix
            @test !(snap1 == snap2)
        end

        @testset "Type Consistency" begin
            dims_RZ = (3, 4)
            snap_f64 = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            snap_f32 = Snapshot2D{Float32}(dims_RZ = dims_RZ)

            # Set some values
            snap_f64.ne[1,1] = 1.5e19
            snap_f32.ne[1,1] = 1.5f19

            # Different float types should be comparable
            @test !(snap_f64 == snap_f32)
            @test !isequal(snap_f64, snap_f32)
            @test isapprox(snap_f64, snap_f32)
        end
    end

    @testset "Cross-Type Comparisons" begin
        # Test that comparing different snapshot types fails gracefully
        snap0D = Snapshot0D{Float64}()
        dims_RZ = (2, 3)
        snap2D = Snapshot2D{Float64}(dims_RZ = dims_RZ)

        # These should not be equal (and shouldn't crash)
        @test !( snap0D == snap2D ) # Fall back to default equality
        @test !isequal(snap0D, snap2D) # Fall back to default equality
        @test_throws MethodError isapprox(snap0D, snap2D)
    end
end
