using RAPID2D
using Test
# using Tempfile

@testset "ADIOS I/O Round-trip Tests" begin


	# Generate test data
	FT = Float64
	# Create simulation configuration
	config = SimulationConfig{FT}(
		NR=10, NZ=20,
		R_min=0.8, R_max=2.2,
		Z_min=-1.2, Z_max=1.2,
		dt=1e-6, t_end_s=10e-6,
		R0B0=1.0,
		prefilled_gas_pressure=5e-3,
		wall_R=[1.0, 2.0, 2.0, 1.0],
		wall_Z=[-1.0, -1.0, 1.0, 1.0]
	)

	# Create RAPID object
	RP = RAPID{FT}(config)

	RP.flags = SimulationFlags{FT}(
		# Disable unnecessary flags for this test
		Ampere=false,
		E_para_self_ES=false, E_para_self_EM=false, Gas_evolve=false,
		update_ni_independently=false, Include_ud_convec_term=false,
		Coulomb_Collision=false, negative_n_correction=false
	)

	# Initial conditions: Gaussian electron density distribution centered in domain
	R0 = (config.R_min + config.R_max) / 2
	Z0 = (config.Z_min + config.Z_max) / 2
	sigma_R = 0.1
	sigma_Z = 0.1
	peak_density = 1.0e6  # Peak density [m^-3]

	initialize!(RP)
	# Initialize electron density
	ini_n = zeros(FT, RP.G.NR, RP.G.NZ)
	for i in 1:RP.G.NR, j in 1:RP.G.NZ
		R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
		ini_n[i, j] = peak_density * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
	end
	ini_n[RP.G.nodes.out_wall_nids] .= 0.0

	RP.plasma.ne = copy(ini_n)
	RP.plasma.ni = copy(ini_n)
	RP.fields.BR_ext .= 1e-4
	RP.fields.BZ_ext .= 1e-4
	RAPID2D.combine_external_and_self_fields!(RP)

	initialize!(RP)
	RAPID2D.run_simulation!(RP);



    @testset "Snapshot0D ADIOS Round-trip" begin
        @testset "Basic Snapshot0D Round-trip" begin
            # Create a test Snapshot0D with realistic plasma parameters

			original_snap = deepcopy(RP.diagnostics.snaps0D[1])  # Use the first snapshot as a template
            # Test round-trip through ADIOS BP format
			tmpdir = mktempdir()
            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap0d.bp")

                # Write to ADIOS BP file
                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                @test isdir(bp_file)

                # Read back using new optimized function
                restored_snaps = adiosBP_to_snap0D(bp_file)

                # Should get back exactly one snapshot
                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Test equality using isequal (NaN-safe)
                @test isequal(original_snap, restored_snap)

                # Test approximate equality as well
                @test isapprox(original_snap, restored_snap; nans=true)

                # Test individual fields for detailed verification
                @test original_snap.ne == restored_snap.ne
                @test original_snap.Te_eV == restored_snap.Te_eV
                @test original_snap.Ti_eV == restored_snap.Ti_eV
                @test original_snap.step == restored_snap.step
                @test original_snap.time_s == restored_snap.time_s
                @test original_snap.dt == restored_snap.dt

                # Test dictionary equality
                @test isequal(original_snap.CFL, restored_snap.CFL)

                # Test array equality
                @test isequal(original_snap.I_coils, restored_snap.I_coils)
            end
        end

        @testset "Snapshot0D with NaN values" begin
            # Test with NaN values to ensure proper handling
            original_snap = Snapshot0D{Float64}()
            original_snap.ne = 1.2e19
            original_snap.Te_eV = NaN          # NaN temperature
            original_snap.Ti_eV = 1.5
            original_snap.step = 500
            original_snap.time_s = 0.01

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap0d_nan.bp")

                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                restored_snaps = adiosBP_to_snap0D(bp_file)

                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Use isequal for NaN-safe comparison
                @test isequal(original_snap, restored_snap)

                # Verify NaN handling specifically
                @test isnan(restored_snap.Te_eV)
                @test original_snap.ne == restored_snap.ne
                @test original_snap.Ti_eV == restored_snap.Ti_eV
            end
        end

        @testset "Multiple Snapshot0D time series" begin
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

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap0d_series.bp")

                @test_nowarn write_to_adiosBP!(bp_file, original_snaps)
                restored_snaps = adiosBP_to_snap0D(bp_file)

                @test length(restored_snaps) == 5

                # Test each snapshot in the series
                for i in 1:5
                    @test isequal(original_snaps[i], restored_snaps[i])
                end
            end
        end
    end

    @testset "Snapshot2D ADIOS Round-trip" begin
        @testset "Basic Snapshot2D Round-trip" begin
            # Create a test Snapshot2D with realistic 2D plasma data
            dims_RZ = (12, 16)  # Small grid for testing
            original_snap = Snapshot2D{Float64}(dims_RZ = dims_RZ)

            # Fill with realistic plasma profiles
            R_axis, Z_axis = size(original_snap.ne)
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


            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d.bp")

                # Write to ADIOS BP file
                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                @test isdir(bp_file)

                # Read back using new optimized function
                restored_snaps = adiosBP_to_snap2D(bp_file)

                # Should get back exactly one snapshot
                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Test equality using isequal (NaN-safe)
                @test isequal(original_snap, restored_snap)

                # Test approximate equality
                @test isapprox(original_snap, restored_snap)

                # Test matrix dimensions
                @test size(original_snap.ne) == size(restored_snap.ne)
                @test size(original_snap.Te_eV) == size(restored_snap.Te_eV)

                # Test scalar fields
                @test original_snap.step == restored_snap.step
                @test original_snap.time_s == restored_snap.time_s
                @test original_snap.dt == restored_snap.dt

                # Test some specific matrix elements
                @test original_snap.ne[1,1] == restored_snap.ne[1,1]
                @test original_snap.ne[6,8] == restored_snap.ne[6,8]
                @test original_snap.Te_eV[3,4] == restored_snap.Te_eV[3,4]

            end
        end

        @testset "Snapshot2D with NaN and special values" begin
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

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d_special.bp")

                @test_nowarn write_to_adiosBP!(bp_file, [original_snap])
                restored_snaps = adiosBP_to_snap2D(bp_file)

                @test length(restored_snaps) == 1
                restored_snap = restored_snaps[1]

                # Use isequal for NaN-safe comparison
                @test isequal(original_snap, restored_snap)

                # Verify special values specifically
                @test isnan(restored_snap.ne[1,1])
                @test isinf(restored_snap.Te_eV[2,2]) && restored_snap.Te_eV[2,2] > 0
                @test isinf(restored_snap.Ti_eV[3,3]) && restored_snap.Ti_eV[3,3] < 0
            end
        end

        @testset "Multiple Snapshot2D time series" begin
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

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_snap2d_series.bp")

                @test_nowarn write_to_adiosBP!(bp_file, original_snaps)
                restored_snaps = adiosBP_to_snap2D(bp_file)

                @test length(restored_snaps) == 3

                # Test each snapshot in the series
                for t in 1:3
                    @test isequal(original_snaps[t], restored_snaps[t])
                end
            end
        end
    end


	@testset "Non-existent file handling" begin
		# Test error handling for non-existent files
		@test_throws Exception adiosBP_to_snap0D("non_existent_file.bp")
		@test_throws Exception adiosBP_to_snap2D("non_existent_file.bp")
	end

    @testset "Performance and Dictionary Filtering" begin


        @testset "Performance comparison hint" begin
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

            mktempdir() do tmpdir
                bp_file = joinpath(tmpdir, "test_performance.bp")

                # Write multiple snapshots
                @test_nowarn write_to_adiosBP!(bp_file, snaps)

                # Read back - should be fast with symbol-based access
                @time restored_snaps = adiosBP_to_snap2D(bp_file)

                @test length(restored_snaps) == 10
                @test isequal(snaps, restored_snaps)
            end
        end
    end
end
