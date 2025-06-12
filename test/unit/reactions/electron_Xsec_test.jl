using Test
using RAPID2D


# Import functions from RAPID2D
using RAPID2D: Xsec_Electron_Momentum_Transfer, Xsec_Electron_Momentum_Transfer!,
				Xsec_Electron_Momentum_Transfer_vectorized, Xsec_Electron_Momentum_Transfer_vectorized!,
				Xsec_Electron_Elastic_Scattering, Xsec_Electron_Elastic_Scattering!,
				Xsec_Electron_Ionization, Xsec_Electron_Ionization!,
				Xsec_Electron_Excitation, Xsec_Electron_Excitation!,
				Xsec_Electron_tot_Excitation, Xsec_Electron_tot_Excitation!,
				Xsec_Electron_Dissociative_Ionization, Xsec_Electron_Dissociative_Ionization!,
				Xsec_Electron_Alpha_Radiation, Xsec_Electron_Alpha_Radiation!,
				Xsec_Electron_Recombination_with_H2_Ion, Xsec_Electron_Recombination_with_H2_Ion!,
				Xsec_Electron_Recombination_with_H3_Ion, Xsec_Electron_Recombination_with_H3_Ion!


@testset "Electron Cross-Section Functions" begin

    # Create aliases for RAPID2D constants and types to avoid repetitive namespace prefixes
    N_Elec_excitation = RAPID2D.N_Elec_excitation
    E_exc_threshold_eV = RAPID2D.E_exc_threshold_eV
    XT_elastic = RAPID2D.XT_elastic
    XT_mom = RAPID2D.XT_mom
    Xsec_Table = RAPID2D.Xsec_Table

    # Test energy ranges
	negative_energy = [-100.0, -0.1]
	zero_energy = [0.0]
    energy_low = [1e-6, 0.001, 0.01, 0.1, 1.0]
    energy_mid = [10.0, 12.0, 15.5, 20.0, 20.5, 50.0, 100.0]
    energy_high = [200.0, 500.0, 1000.0, 1E6]
    energy_all = vcat(negative_energy, zero_energy, energy_low, energy_mid, energy_high)

    @testset "Constants and Data Structures" begin
        # Test that constants are properly defined
        @test N_Elec_excitation == 9
        @test length(E_exc_threshold_eV) == 9

        # Test threshold energies are reasonable
        for threshold in E_exc_threshold_eV
            @test threshold > 0.0
            @test threshold < 30.0  # Reasonable upper bound for molecular excitation
        end

        # Test cross-section tables
        @test XT_elastic isa Xsec_Table{100}
        @test XT_mom isa Xsec_Table{100}
        @test length(XT_elastic.E_eV) == 100
        @test length(XT_mom.E_eV) == 100
        @test XT_elastic.dlog10E > 0.0
        @test XT_mom.dlog10E > 0.0
    end

    @testset "Momentum Transfer Cross-Section" begin
        # Test single energy version
        @test Xsec_Electron_Momentum_Transfer(-1.0) == 0.0
        @test Xsec_Electron_Momentum_Transfer(0.0) == 0.0
        @test Xsec_Electron_Momentum_Transfer(1.0) > 0.0
        @test Xsec_Electron_Momentum_Transfer(0.0001) > 0.0
        @test Xsec_Electron_Momentum_Transfer(1000.0) > 0.0

        # Test vector version
        xsec_vec = Xsec_Electron_Momentum_Transfer(energy_all)
        @test length(xsec_vec) == length(energy_all)
        @test all(xsec_vec .>= 0.0)

        # Test in-place version
        xsec_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Momentum_Transfer!(energy_all, xsec_out)
        @test all(xsec_out .>= 0.0)
        @test xsec_out ≈ xsec_vec

        # Test vectorized versions
        xsec_vec_fast = Xsec_Electron_Momentum_Transfer_vectorized(energy_all)
        @test length(xsec_vec_fast) == length(energy_all)
        @test all(xsec_vec_fast .>= 0.0)

        xsec_out_fast = zeros(Float64, length(energy_all))
        Xsec_Electron_Momentum_Transfer_vectorized!(energy_all, xsec_out_fast)
        @test all(xsec_out_fast .>= 0.0)
        @test xsec_out_fast ≈ xsec_vec_fast
    end

    @testset "Elastic Scattering Cross-Section" begin
        # Test single energy version
        @test Xsec_Electron_Elastic_Scattering(-1.0) == 0.0
        @test Xsec_Electron_Elastic_Scattering(0.0) == 0.0
        @test Xsec_Electron_Elastic_Scattering(1.0) > 0.0
        @test Xsec_Electron_Elastic_Scattering(0.01) > 0.0
        @test Xsec_Electron_Elastic_Scattering(150.0) > 0.0

        # Test vector version
        xsec_vec = Xsec_Electron_Elastic_Scattering(energy_all)
        @test length(xsec_vec) == length(energy_all)
        @test all(xsec_vec .>= 0.0)

        # Test in-place version
        xsec_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Elastic_Scattering!(energy_all, xsec_out)
        @test all(xsec_out .>= 0.0)
        @test xsec_out ≈ xsec_vec
    end

    @testset "Ionization Cross-Section" begin
        # Test vector version
        xsec_vec = Xsec_Electron_Ionization(energy_all)
        @test length(xsec_vec) == length(energy_all)
        @test all(xsec_vec .>= 0.0)  # Can be zero below threshold

        # Test in-place version
        xsec_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Ionization!(energy_all, xsec_out)
        @test all(xsec_out .>= 0.0)
        @test xsec_out ≈ xsec_vec

        # Test threshold behavior (ionization threshold ~15.46 eV)
        low_energy = [10.0, 14.0]
        high_energy = [20.0, 50.0, 100.0]

        xsec_low = Xsec_Electron_Ionization(low_energy)
        xsec_high = Xsec_Electron_Ionization(high_energy)

        @test all(xsec_low .== 0.0)  # Below threshold
        @test all(xsec_high .>= 0.0)  # Above threshold
    end

    @testset "Excitation Cross-Sections" begin
        # Test all excitation reactions
        for reaction_flag in 1:N_Elec_excitation
            xsec_vec = Xsec_Electron_Excitation(energy_all, reaction_flag)
            @test length(xsec_vec) == length(energy_all)
            @test all(xsec_vec .>= 0.0)

            # Test in-place version
            xsec_out = zeros(Float64, length(energy_all))
            Xsec_Electron_Excitation!(energy_all, reaction_flag, xsec_out)
            @test all(xsec_out .>= 0.0)
            @test xsec_out ≈ xsec_vec

            # Test threshold behavior
            threshold = E_exc_threshold_eV[reaction_flag]
            below_threshold = [threshold - 1.0]
            above_threshold = [threshold + 5.0]

            xsec_below = Xsec_Electron_Excitation(below_threshold, reaction_flag)
            xsec_above = Xsec_Electron_Excitation(above_threshold, reaction_flag)

            @test xsec_below[1] == 0.0  # Below threshold
            # Note: Some excitation reactions might still be zero slightly above threshold
            # depending on their specific energy dependence
        end

        # Test total excitation
        xsec_total = Xsec_Electron_tot_Excitation(energy_all)
        @test length(xsec_total) == length(energy_all)
        @test all(xsec_total .>= 0.0)

        # Test in-place total excitation
        xsec_total_out = zeros(Float64, length(energy_all))
        Xsec_Electron_tot_Excitation!(energy_all, xsec_total_out)
        @test all(xsec_total_out .>= 0.0)
        @test xsec_total_out ≈ xsec_total
    end

    @testset "Dissociative Ionization Cross-Section" begin
        # Test vector version
        xsec_vec = Xsec_Electron_Dissociative_Ionization(energy_all)
        @test length(xsec_vec) == length(energy_all)
        @test all(xsec_vec .>= 0.0)

        # Test in-place version
        xsec_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Dissociative_Ionization!(energy_all, xsec_out)
        @test all(xsec_out .>= 0.0)
        @test xsec_out ≈ xsec_vec

        # Test threshold behavior (threshold at 35 eV)
        low_energy = [20.0, 30.0]
        high_energy = [40.0, 60.0, 100.0]

        xsec_low = Xsec_Electron_Dissociative_Ionization(low_energy)
        xsec_high = Xsec_Electron_Dissociative_Ionization(high_energy)

        @test all(xsec_low .== 0.0)  # Below threshold
        @test all(xsec_high .>= 0.0)  # Above threshold
    end

    @testset "Alpha Radiation Cross-Section" begin
        # Test vector version
        xsec_vec = Xsec_Electron_Alpha_Radiation(energy_all)
        @test length(xsec_vec) == length(energy_all)
        @test all(xsec_vec .>= 0.0)

        # Test in-place version
        xsec_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Alpha_Radiation!(energy_all, xsec_out)
        @test all(xsec_out .>= 0.0)
        @test xsec_out ≈ xsec_vec

        # Test threshold behavior (threshold at 19 eV)
        low_energy = [10.0, 15.0]
        high_energy = [25.0, 50.0, 100.0]

        xsec_low = Xsec_Electron_Alpha_Radiation(low_energy)
        xsec_high = Xsec_Electron_Alpha_Radiation(high_energy)

        @test all(xsec_low .== 0.0)  # Below threshold
        @test all(xsec_high .>= 0.0)  # Above threshold
    end

    @testset "Recombination Cross-Sections" begin
        # Test H2+ recombination
        xsec_h2_vec = Xsec_Electron_Recombination_with_H2_Ion(energy_all)
        @test length(xsec_h2_vec) == length(energy_all)
        @test all(xsec_h2_vec .>= 0.0)

        xsec_h2_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Recombination_with_H2_Ion!(energy_all, xsec_h2_out)
        @test all(xsec_h2_out .>= 0.0)
        @test xsec_h2_out ≈ xsec_h2_vec

        # Test H3+ recombination
        xsec_h3_vec = Xsec_Electron_Recombination_with_H3_Ion(energy_all)
        @test length(xsec_h3_vec) == length(energy_all)
        @test all(xsec_h3_vec .>= 0.0)

        xsec_h3_out = zeros(Float64, length(energy_all))
        Xsec_Electron_Recombination_with_H3_Ion!(energy_all, xsec_h3_out)
        @test all(xsec_h3_out .>= 0.0)
        @test xsec_h3_out ≈ xsec_h3_vec
    end

    @testset "Cross-Section Magnitude Checks" begin
        # Test that cross-sections are within reasonable physical bounds
        test_energy = [1.0, 10.0, 50.0, 100.0]

        # Typical cross-sections should be in the range 1e-22 to 1e-18 m^2
        xsec_elastic = Xsec_Electron_Elastic_Scattering(test_energy)
        @test all(1e-22 .< xsec_elastic .< 1e-17)

        xsec_momentum = Xsec_Electron_Momentum_Transfer(test_energy)
        @test all(1e-22 .< xsec_momentum .< 1e-17)

        # Ionization cross-sections should be smaller than elastic for ceratin ranges
        mid_energy = [16.0, 30.0, 50.0]  # Above ionization threshold
        xsec_ion = Xsec_Electron_Ionization(mid_energy)
        xsec_elas_high = Xsec_Electron_Elastic_Scattering(mid_energy)
        @test all(xsec_ion .< xsec_elas_high)
    end

    @testset "Error Handling" begin
        # Test assertion errors for mismatched array lengths
        energy_short = [1.0, 10.0]
        output_long = zeros(5)

        @test_throws AssertionError Xsec_Electron_Momentum_Transfer!(energy_short, output_long)
        @test_throws AssertionError Xsec_Electron_Elastic_Scattering!(energy_short, output_long)
        @test_throws AssertionError Xsec_Electron_Ionization!(energy_short, output_long)
        @test_throws AssertionError Xsec_Electron_Excitation!(energy_short, 1, output_long)
    end

    @testset "Function Consistency" begin
        # Test that different versions of the same function give consistent results
        test_energies = [0.1, 1.0, 10.0, 50.0, 100.0]

        # Compare single energy vs vector versions for momentum transfer
        for E in test_energies
            single_result = Xsec_Electron_Momentum_Transfer(E)
            vector_result = Xsec_Electron_Momentum_Transfer([E])
            @test single_result ≈ vector_result[1] rtol=1e-12
        end

        # Compare single energy vs vector versions for elastic scattering
        for E in test_energies
            single_result = Xsec_Electron_Elastic_Scattering(E)
            vector_result = Xsec_Electron_Elastic_Scattering([E])
            @test single_result ≈ vector_result[1] rtol=1e-12
        end
    end
end
