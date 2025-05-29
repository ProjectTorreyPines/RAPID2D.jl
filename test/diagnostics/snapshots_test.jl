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
