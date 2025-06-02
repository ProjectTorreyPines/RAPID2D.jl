using RAPID2D
using Test

@testset "ADIOS I/O Round-trip Tests" begin
    println("DEBUG: Starting ADIOS I/O Round-trip Tests")
    flush(stdout)

    @testset "Snapshot0D ADIOS Round-trip" begin
        println("DEBUG: Starting Snapshot0D ADIOS Round-trip tests")
        flush(stdout)

        @testset "Basic Snapshot0D Round-trip" begin
            println("DEBUG: Starting Basic Snapshot0D Round-trip test")
            flush(stdout)

            # Create a test Snapshot0D with realistic plasma parameters
			println("DEBUG: Creating Snapshot0D objects")
			flush(stdout)
			original_snap = Snapshot0D{Float64}()
			snap2 = Snapshot0D{Float64}()

			println("DEBUG: Setting Snapshot0D parameters")
			flush(stdout)
			original_snap.ne = 1.5e19
			original_snap.Te_eV = 5.5
			original_snap.step = 100

			# original_snap = deepcopy(RP.diagnostics.snaps0D[1])  # Use the first snapshot as a template
            # Test round-trip through ADIOS BP format
			println("DEBUG: Creating temporary directory")
			flush(stdout)
			tmpdir = mktempdir()
            mktempdir() do tmpdir
                println("DEBUG: Created tmpdir: $tmpdir")
                flush(stdout)
                bp_file = joinpath(tmpdir, "test_snap0d.bp")
                println("DEBUG: BP file path: $bp_file")
                flush(stdout)

                # Write to ADIOS BP file
                println("DEBUG: About to write to ADIOS BP file")
                flush(stdout)
                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                println("DEBUG: Successfully wrote to ADIOS BP file")
                flush(stdout)

                @test isdir(bp_file)
                println("DEBUG: Verified BP file directory exists")
                flush(stdout)

                # Read back using new optimized function
                println("DEBUG: About to read back from ADIOS BP file")
                flush(stdout)
                restored_snaps = adiosBP_to_snap0D(bp_file)
                println("DEBUG: Successfully read from ADIOS BP file")
                flush(stdout)

                # Should get back exactly one snapshot
                println("DEBUG: Checking number of restored snapshots")
                flush(stdout)
                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]
                println("DEBUG: Successfully retrieved restored snapshot")
                flush(stdout)

                # Test equality using isequal (NaN-safe)
                println("DEBUG: Testing snapshot equality")
                flush(stdout)
                @test isequal(original_snap, restored_snap)

                # Test approximate equality as well
                println("DEBUG: Testing approximate equality")
                flush(stdout)
                @test isapprox(original_snap, restored_snap; nans=true)

                # Test individual fields for detailed verification
                println("DEBUG: Testing individual fields")
                flush(stdout)
                @test original_snap.ne == restored_snap.ne
                @test original_snap.Te_eV == restored_snap.Te_eV
                @test original_snap.Ti_eV == restored_snap.Ti_eV
                @test original_snap.step == restored_snap.step
                @test original_snap.time_s == restored_snap.time_s
                @test original_snap.dt == restored_snap.dt

                # Test dictionary equality
                println("DEBUG: Testing dictionary equality")
                flush(stdout)
                @test isequal(original_snap.CFL, restored_snap.CFL)

                # Test array equality
                println("DEBUG: Testing array equality")
                flush(stdout)
                @test isequal(original_snap.I_coils, restored_snap.I_coils)

                println("DEBUG: Basic Snapshot0D Round-trip test completed successfully")
                flush(stdout)
            end
        end

        @testset "Snapshot0D with NaN values" begin
            println("DEBUG: Starting Snapshot0D with NaN values test")
            flush(stdout)

            # Test with NaN values to ensure proper handling
            original_snap = Snapshot0D{Float64}()
            original_snap.ne = 1.2e19
            original_snap.Te_eV = NaN          # NaN temperature
            original_snap.Ti_eV = 1.5
            original_snap.step = 500
            original_snap.time_s = 0.01
            println("DEBUG: Created Snapshot0D with NaN values")
            flush(stdout)

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap0d_nan.bp")
                println("DEBUG: About to write NaN test to: $bp_file")
                flush(stdout)

                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                println("DEBUG: Successfully wrote NaN test to ADIOS")
                flush(stdout)

                restored_snaps = adiosBP_to_snap0D(bp_file)
                println("DEBUG: Successfully read NaN test from ADIOS")
                flush(stdout)

                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Use isequal for NaN-safe comparison
                @test isequal(original_snap, restored_snap)

                # Verify NaN handling specifically
                @test isnan(restored_snap.Te_eV)
                @test original_snap.ne == restored_snap.ne
                @test original_snap.Ti_eV == restored_snap.Ti_eV

                println("DEBUG: NaN test completed successfully")
                flush(stdout)
            end
        end

        @testset "Multiple Snapshot0D time series" begin
            println("DEBUG: Starting Multiple Snapshot0D time series test")
            flush(stdout)

            # Create a time series of snapshots
            original_snaps = Snapshot0D{Float64}[]
            for i in 1:5
                snap = Snapshot0D{Float64}()
                snap.ne = 1.0e19 + i * 0.1e19
                snap.Te_eV = 5.0 + i * 0.5
                snap.Ti_eV = 2.0 + i * 0.2
                snap.step = i * 100
                snap.time_s = i * 0.005
                push!(original_snaps, snap)
            end
            println("DEBUG: Created time series of 5 snapshots")
            flush(stdout)

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap0d_series.bp")
                println("DEBUG: About to write time series to: $bp_file")
                flush(stdout)

                @test_nowarn write_to_adiosBP!(bp_file, original_snaps)
                println("DEBUG: Successfully wrote time series to ADIOS")
                flush(stdout)

                restored_snaps = adiosBP_to_snap0D(bp_file)
                println("DEBUG: Successfully read time series from ADIOS")
                flush(stdout)

                @test length(restored_snaps) == 5

                # Test each snapshot in the series
                for i in 1:5
                    @test isequal(original_snaps[i], restored_snaps[i])
                end

                println("DEBUG: Time series test completed successfully")
                flush(stdout)
            end
        end
    end

    @testset "Snapshot2D ADIOS Round-trip" begin
        println("DEBUG: Starting Snapshot2D ADIOS Round-trip tests")
        flush(stdout)

        @testset "Basic Snapshot2D Round-trip" begin
            println("DEBUG: Starting Basic Snapshot2D Round-trip test")
            flush(stdout)

            # Create a test Snapshot2D with realistic 2D plasma data
            dims_RZ = (12, 16)  # Small grid for testing
            println("DEBUG: Creating Snapshot2D with dimensions: $dims_RZ")
            flush(stdout)
            original_snap = Snapshot2D{Float64}(dims_RZ = dims_RZ)
            println("DEBUG: Created Snapshot2D object")
            flush(stdout)

            # Fill with realistic plasma profiles
            R_axis, Z_axis = size(original_snap.ne)
            println("DEBUG: Filling Snapshot2D with plasma profiles")
            flush(stdout)
            for i in 1:R_axis, j in 1:Z_axis
                # Create radial profiles (higher density/temperature in center)
                r_norm = sqrt((i - R_axis/2)^2 + (j - Z_axis/2)^2) / (R_axis/2)
                profile_factor = exp(-r_norm^2 * 2)  # Gaussian-like profile

                original_snap.ne[i,j] = 2.0e19 * profile_factor
                original_snap.Te_eV[i,j] = 10.0 * profile_factor + 1.0  # Core: 11 eV, edge: 1 eV
                original_snap.Ti_eV[i,j] = 8.0 * profile_factor + 0.5   # Core: 8.5 eV, edge: 0.5 eV
            end

            # Set scalar fields
            original_snap.step = 2000
            original_snap.time_s = 0.05
            original_snap.dt = 2.5e-5
            println("DEBUG: Finished filling Snapshot2D data")
            flush(stdout)


            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d.bp")
                println("DEBUG: About to write Snapshot2D to: $bp_file")
                flush(stdout)

                # Write to ADIOS BP file
                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                println("DEBUG: Successfully wrote Snapshot2D to ADIOS")
                flush(stdout)

                @test isdir(bp_file)
                println("DEBUG: Verified Snapshot2D BP file directory exists")
                flush(stdout)

                # Read back using new optimized function
                println("DEBUG: About to read Snapshot2D from ADIOS")
                flush(stdout)
                restored_snaps = adiosBP_to_snap2D(bp_file)
                println("DEBUG: Successfully read Snapshot2D from ADIOS")
                flush(stdout)

                # Should get back exactly one snapshot
                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]
                println("DEBUG: Retrieved restored Snapshot2D")
                flush(stdout)

                # Test equality using isequal (NaN-safe)
                println("DEBUG: Testing Snapshot2D equality")
                flush(stdout)
                @test isequal(original_snap, restored_snap)

                # Test approximate equality
                println("DEBUG: Testing Snapshot2D approximate equality")
                flush(stdout)
                @test isapprox(original_snap, restored_snap)

                # Test matrix dimensions
                println("DEBUG: Testing Snapshot2D matrix dimensions")
                flush(stdout)
                @test size(original_snap.ne) == size(restored_snap.ne)
                @test size(original_snap.Te_eV) == size(restored_snap.Te_eV)

                # Test scalar fields
                println("DEBUG: Testing Snapshot2D scalar fields")
                flush(stdout)
                @test original_snap.step == restored_snap.step
                @test original_snap.time_s == restored_snap.time_s
                @test original_snap.dt == restored_snap.dt

                # Test some specific matrix elements
                println("DEBUG: Testing Snapshot2D specific matrix elements")
                flush(stdout)
                @test original_snap.ne[1,1] == restored_snap.ne[1,1]
                @test original_snap.ne[6,8] == restored_snap.ne[6,8]
                @test original_snap.Te_eV[3,4] == restored_snap.Te_eV[3,4]

                println("DEBUG: Basic Snapshot2D Round-trip test completed successfully")
                flush(stdout)

            end
        end

        @testset "Snapshot2D with NaN and special values" begin
            println("DEBUG: Starting Snapshot2D with NaN and special values test")
            flush(stdout)

            dims_RZ = (8, 10)
            original_snap = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Fill most with normal values
            fill!(original_snap.ne, 1.5e19)
            fill!(original_snap.Te_eV, 5.0)

            # Add some special values
            original_snap.ne[1,1] = NaN
            original_snap.Te_eV[2,2] = Inf
            original_snap.Ti_eV[3,3] = -Inf

            original_snap.step = 1500
            original_snap.time_s = 0.03
            println("DEBUG: Created Snapshot2D with special values")
            flush(stdout)

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d_special.bp")
                println("DEBUG: About to write Snapshot2D special values test to: $bp_file")
                flush(stdout)

                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                println("DEBUG: Successfully wrote Snapshot2D special values to ADIOS")
                flush(stdout)

                restored_snaps = adiosBP_to_snap2D(bp_file)
                println("DEBUG: Successfully read Snapshot2D special values from ADIOS")
                flush(stdout)

                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Use isequal for NaN-safe comparison
                @test isequal(original_snap, restored_snap)

                # Verify special values specifically
                @test isnan(restored_snap.ne[1,1])
                @test isinf(restored_snap.Te_eV[2,2]) && restored_snap.Te_eV[2,2] > 0
                @test isinf(restored_snap.Ti_eV[3,3]) && restored_snap.Ti_eV[3,3] < 0

                println("DEBUG: Snapshot2D special values test completed successfully")
                flush(stdout)
            end
        end

        @testset "Multiple Snapshot2D time series" begin
            println("DEBUG: Starting Multiple Snapshot2D time series test")
            flush(stdout)

            dims_RZ = (6, 8)
            original_snaps = Snapshot2D{Float64}[]

            for t in 1:3
                snap = Snapshot2D{Float64}(dims_RZ = dims_RZ)

                # Create time-evolving profiles
                for i in 1:dims_RZ[1], j in 1:dims_RZ[2]
                    snap.ne[i,j] = 1.0e19 * (1.0 + 0.1 * t)
                    snap.Te_eV[i,j] = 5.0 + t * 0.5
                end

                snap.step = t * 500
                snap.time_s = t * 0.01

                push!(original_snaps, snap)
            end
            println("DEBUG: Created Snapshot2D time series of 3 snapshots")
            flush(stdout)

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d_series.bp")
                println("DEBUG: About to write Snapshot2D time series to: $bp_file")
                flush(stdout)

                @test_nowarn write_to_adiosBP!(bp_file, original_snaps)
                println("DEBUG: Successfully wrote Snapshot2D time series to ADIOS")
                flush(stdout)

                restored_snaps = adiosBP_to_snap2D(bp_file)
                println("DEBUG: Successfully read Snapshot2D time series from ADIOS")
                flush(stdout)

                @test length(restored_snaps) == 3

                # Test each snapshot in the series
                for t in 1:3
                    @test isequal(original_snaps[t], restored_snaps[t])
                end

                println("DEBUG: Snapshot2D time series test completed successfully")
                flush(stdout)
            end
        end
    end


	@testset "Non-existent file handling" begin
		println("DEBUG: Starting Non-existent file handling test")
		flush(stdout)

		# Test error handling for non-existent files
		println("DEBUG: Testing error handling for non-existent 0D file")
		flush(stdout)
		@test_throws Exception adiosBP_to_snap0D("non_existent_file.bp")

		println("DEBUG: Testing error handling for non-existent 2D file")
		flush(stdout)
		@test_throws Exception adiosBP_to_snap2D("non_existent_file.bp")

		println("DEBUG: Non-existent file handling test completed")
		flush(stdout)
	end

    @testset "Performance and Dictionary Filtering" begin
        println("DEBUG: Starting Performance and Dictionary Filtering tests")
        flush(stdout)


        @testset "Performance comparison hint" begin
            println("DEBUG: Starting Performance comparison hint test")
            flush(stdout)

            # This test doesn't actually measure performance, but documents
            # the expected behavior for future performance testing
            dims_RZ = (20, 25)  # Larger grid
            snaps = Snapshot2D{Float64}[]

            for i in 1:10  # Multiple time steps
                snap = Snapshot2D{Float64}(dims_RZ = dims_RZ)
                fill!(snap.ne, 1.0e19 + i * 0.1e19)
                fill!(snap.Te_eV, 5.0 + i * 0.1)
                snap.step = i * 100
                push!(snaps, snap)
            end
            println("DEBUG: Created performance test with 10 large snapshots")
            flush(stdout)

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_performance.bp")
                println("DEBUG: About to write performance test to: $bp_file")
                flush(stdout)

                # Write multiple snapshots
                @test_nowarn write_to_adiosBP!(bp_file, snaps)
                println("DEBUG: Successfully wrote performance test to ADIOS")
                flush(stdout)

                # Read back - should be fast with symbol-based access
                println("DEBUG: About to read performance test from ADIOS")
                flush(stdout)
                @time restored_snaps = adiosBP_to_snap2D(bp_file)
                println("DEBUG: Successfully read performance test from ADIOS")
                flush(stdout)

                @test length(restored_snaps) == 10
                @test isequal(snaps, restored_snaps)

                println("DEBUG: Performance comparison hint test completed successfully")
                flush(stdout)
            end
        end
    end

    println("DEBUG: All ADIOS I/O Round-trip Tests completed successfully")
    flush(stdout)
end
