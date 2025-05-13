using Test
using HDF5
using Interpolations
using RAPID2D

@testset "Reaction Rate Coefficients" begin
    # Define paths to test data
    RRC_data_dir = joinpath(dirname(dirname(@__DIR__)), "RRC_data")
    EoverP_Erg_file = joinpath(RRC_data_dir, "eRRCs_EoverP_Erg.h5")
    e_tud_file = joinpath(RRC_data_dir, "eRRCs_T_ud.h5")
    i_tud_file = joinpath(RRC_data_dir, "iRRCs_T_ud.h5")

    @testset "File Access" begin
        # Test that files exist
        @test isfile(EoverP_Erg_file)
        @test isfile(e_tud_file)
        @test isfile(i_tud_file)

        # Test we can open and read from files
        h5open(EoverP_Erg_file, "r") do file
            @test haskey(file, "EoverP")
            @test haskey(file, "Erg_eV")
        end

        h5open(e_tud_file, "r") do file
            @test haskey(file, "T_eV")
            @test haskey(file, "ud_para")
        end

        h5open(i_tud_file, "r") do file
            @test haskey(file, "T_eV")
            @test haskey(file, "ud_para")
        end
    end

    @testset "RRC_EoverP_Erg" begin
        # Load test data directly
        h5open(EoverP_Erg_file, "r") do file
            EoverP = read(file, "EoverP")
            Erg_eV = read(file, "Erg_eV")
            ionization_data = read(file, "Ionization")

            # Test constructor
            rrc = RRC_EoverP_Erg(EoverP, Erg_eV, ionization_data)

            # Test struct fields
            @test rrc.EoverP == EoverP
            @test rrc.Erg_eV == Erg_eV
            @test rrc.raw_data == ionization_data

            # Test interpolation
            # Pick a point in the middle of the range
            midpoint_eoeverp = EoverP[div(length(EoverP), 2)]
            midpoint_erg = Erg_eV[div(length(Erg_eV), 2)]

            # Interpolation should match the raw data at grid points
            @test rrc.itp(midpoint_eoeverp, midpoint_erg) ≈
                  ionization_data[div(length(EoverP), 2), div(length(Erg_eV), 2)]

            # Test interpolation at point between grid points
            if length(EoverP) > 1 && length(Erg_eV) > 1
                interp_eoeverp = (EoverP[1] + EoverP[2]) / 2
                interp_erg = (Erg_eV[1] + Erg_eV[2]) / 2
                # Just test that it doesn't error
                @test !isnan(rrc.itp(interp_eoeverp, interp_erg))
            end

            # Test bounds handling
            @test_throws BoundsError rrc.itp(EoverP[1] - 1.0, Erg_eV[1])
            @test_throws BoundsError rrc.itp(EoverP[end] + 1.0, Erg_eV[1])
        end
    end

    @testset "RRC_T_ud" begin
        # Load test data directly
        h5open(i_tud_file, "r") do file
            T_eV = read(file, "T_eV")
            ud_para = read(file, "ud_para")
            # Use a reaction that should exist in the file
            elastic_data = read(file, "Elastic")

            # Test constructor
            RRC = RRC_T_ud(T_eV, ud_para, elastic_data)

            # Test struct fields
            @test RRC.T_eV == T_eV
            @test RRC.ud_para == ud_para
            @test RRC.raw_data == elastic_data

            # Test interpolation at grid points
            midpoint_T = T_eV[div(length(T_eV), 2)]
            midpoint_ud = ud_para[div(length(ud_para), 2)]

            @test RRC.itp(midpoint_T, midpoint_ud) ≈
                  elastic_data[div(length(T_eV), 2), div(length(ud_para), 2)]

            # Test interpolation at point between grid points
            if length(T_eV) > 1 && length(ud_para) > 1
                interp_T = (T_eV[1] + T_eV[2]) / 2
                interp_ud = (ud_para[1] + ud_para[2]) / 2
                # Just test that it doesn't error
                @test !isnan(RRC.itp(interp_T, interp_ud))
            end
        end
    end

    @testset "Electron_RRCs" begin
        # Test full container construction
        e_rrcs = Electron_RRCs(EoverP_Erg_file, e_tud_file)

        # Test field types
        @test e_rrcs.Ionization isa RRC_EoverP_Erg
        @test e_rrcs.Momentum isa RRC_EoverP_Erg
        @test e_rrcs.Total_Excitation isa RRC_EoverP_Erg

        # Test data availability
        @test length(e_rrcs.Ionization.EoverP) > 0
        @test length(e_rrcs.Ionization.Erg_eV) > 0
        @test size(e_rrcs.Ionization.raw_data, 1) == length(e_rrcs.Ionization.EoverP)
        @test size(e_rrcs.Ionization.raw_data, 2) == length(e_rrcs.Ionization.Erg_eV)
    end

    @testset "H2_Ion_RRCs" begin
        # Test full container construction
        i_rrcs = H2_Ion_RRCs(i_tud_file)

        # Test field types
        @test i_rrcs.Elastic isa RRC_T_ud
        @test i_rrcs.Charge_Exchange isa RRC_T_ud
        @test i_rrcs.Target_Ionization isa RRC_T_ud

        # Test data availability
        @test length(i_rrcs.Elastic.T_eV) > 0
        @test length(i_rrcs.Elastic.ud_para) > 0
        @test size(i_rrcs.Elastic.raw_data, 1) == length(i_rrcs.Elastic.T_eV)
        @test size(i_rrcs.Elastic.raw_data, 2) == length(i_rrcs.Elastic.ud_para)
    end

    @testset "Sample RRC Calculation" begin
        # Create mock RAPID model for testing get_electron_RRC and get_H2_ion_RRC
        # This is a simplified version for testing only

        # Create minimal RAPID struct for testing
        FT = Float64

        # Create mock RAPID struct
		config =  RAPID2D.SimulationConfig{Float64}()
		config.prefilled_gas_pressure = 4e-3;
		config.R0B0 = 1.5*1.8;

		mock_RP = create_rapid_object(; config=config);

        # Load RRCs
        eRRCs = Electron_RRCs(EoverP_Erg_file, e_tud_file)
        iRRCs = H2_Ion_RRCs(i_tud_file)

		mock_RP.plasma.Te_eV .= 10.0

        # Test get_electron_RRC function - make sure it runs without errors
        # and returns expected size
        ionization_RRC = get_electron_RRC(mock_RP, eRRCs, :Ionization)
        @test size(ionization_RRC) == size(mock_RP.G.R2D)
        @test !any(isnan.(ionization_RRC))

        Halpha_RRC = get_electron_RRC(mock_RP, eRRCs, :Halpha)
        @test size(Halpha_RRC) == size(mock_RP.G.R2D)
        @test !any(isnan.(Halpha_RRC))

        # Test get_H2_ion_RRC function - make sure it runs without errors
        # and returns expected size
        elastic_rates = get_H2_ion_RRC(mock_RP, iRRCs, :Elastic)
        @test size(elastic_rates) == size(mock_RP.G.R2D)
        @test !any(isnan.(elastic_rates))
    end
end