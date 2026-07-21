# Scalar-field gradient helpers and the 2D smoothing kernel. Both testitems build
# and mutate a full RAPID object on a 100x200 grid; the shared SimulationConfig
# factory lives in setup_numerics.jl.

@testitem "Gradient Calculation Tests" setup=[NumericsFixtures] begin

    FT = Float64
    config = walled_box_config(FT)

    RP = RAPID{FT}(config)
    initialize!(RP)

    # Set up a specific siutation for testing
    RP.fields.BR_ext .= 20e-4
    RP.fields.BZ_ext .= 10e-4
    RAPID2D.combine_external_and_self_fields!(RP)

    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.3
    sigma_Z = 0.3
    peak_ue_para = 1.0e6  # Peak velocity
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
        RP.plasma.ue_para[i, j] = peak_ue_para * exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end
    RAPID2D.update_transport_quantities!(RP)


    # Create a known scalar field for testing (e.g., a simple function like z^2 + r^3)
    NR, NZ = RP.G.NR, RP.G.NZ

    F_test = @. RP.G.R2D^3 + RP.G.Z2D^2

    # Known analytic gradients
    expected_∇F_R = @. 3*RP.G.R2D^2
    expected_∇F_Z = 2*RP.G.Z2D

    # Calculate gradients using our function (no upwind)
    ∇F_R, ∇F_Z = RAPID2D.calculate_grad_of_scalar_F(RP, F_test; upwind=false)
    tmp_para_∇F = @. ∇F_R*RP.fields.bR + ∇F_Z*RP.fields.bZ;
    para_∇F = RAPID2D.calculate_para_grad_of_scalar_F(RP, F_test; upwind=false)
    @test isapprox(∇F_R, expected_∇F_R, rtol=1e-2)
    @test isapprox(∇F_Z, expected_∇F_Z, rtol=1e-2)
    @test isapprox(tmp_para_∇F, para_∇F, rtol=1e-10)

    # Calculate gradients using our function (with upwind)
    ∇F_R, ∇F_Z = RAPID2D.calculate_grad_of_scalar_F(RP, F_test; upwind=true)
    tmp_para_∇F = @. ∇F_R*RP.fields.bR + ∇F_Z*RP.fields.bZ;
    para_∇F = RAPID2D.calculate_para_grad_of_scalar_F(RP, F_test; upwind=true)
    @test isapprox(∇F_R, expected_∇F_R, rtol=1e-2)
    @test isapprox(∇F_Z, expected_∇F_Z, rtol=1e-2)
    @test isapprox(tmp_para_∇F, para_∇F, rtol=1e-10)
end



@testitem "smooth_data_2D" setup=[NumericsFixtures] begin

    FT = Float64
    config = walled_box_config(FT)

    RP = RAPID{FT}(config)
    initialize!(RP)

    # Set up a specific ff
    R0 = (config.R_min + config.R_max) / 2
    Z0 = (config.Z_min + config.Z_max) / 2
    sigma_R = 0.3
    sigma_Z = 0.3
    ff = zeros(FT, RP.G.NR, RP.G.NZ)
    for i in 1:RP.G.NR, j in 1:RP.G.NZ
        R, Z= RP.G.R2D[i, j], RP.G.Z2D[i, j]
        ff[i, j] =  exp(-((R-R0)^2/(2*sigma_R^2) + (Z-Z0)^2/(2*sigma_Z^2)))
    end

    # ORDERING IS LOAD-BEARING — DO NOT SPLIT THESE TWO BLOCKS INTO SEPARATE
    # @testitems. `ff` is shared and MUTATED: the "without weighting" block ends
    # with an in-place `smooth_data_2D!(ff; num_SM=3)`, so the "with weighting"
    # block deliberately starts from the ALREADY-SMOOTHED field. Splitting them
    # would hand the second block a pristine Gaussian instead and silently change
    # what is being asserted.
    @testset "smooth_data_2D without weighting" begin
        # No smoothing
        num_SM = 0
        ff_SM = RAPID2D.smooth_data_2D(ff; num_SM)
        @test ff_SM == ff
        RAPID2D.smooth_data_2D!(ff; num_SM)
        @test ff_SM == ff

        # Smooth with num_SM = 3
        num_SM = 3
        ff_SM = RAPID2D.smooth_data_2D(ff; num_SM)
        @test ff_SM ≉ ff
        @test isapprox(sum(ff_SM), sum(ff), rtol=1e-12)
        RAPID2D.smooth_data_2D!(ff; num_SM)
        @test ff_SM == ff
    end

    @testset "smooth_data_2D with weighting" begin
        # `copy` is REQUIRED: `RP.G.Jacob` is a live field of the grid, and the
        # "test with zero weighting" section below mutates `weighting` IN PLACE
        # (`weighting[1, 1:3] .= 0.0`). Binding it directly would alias and
        # silently corrupt the RAPID object's own Jacobian.
        weighting = copy(RP.G.Jacob)
        # No smoothing
        num_SM = 0
        ff_SM = RAPID2D.smooth_data_2D(ff; num_SM, weighting)
        @test ff_SM == ff
        RAPID2D.smooth_data_2D!(ff; num_SM)
        @test ff_SM == ff

        # Smooth with num_SM = 3
        num_SM = 3
        ff_SM = RAPID2D.smooth_data_2D(ff; num_SM, weighting)
        @test ff_SM ≉ ff
        @test !isapprox(sum(ff_SM), sum(ff), rtol=1e-12)
        @test isapprox(sum(ff_SM.*weighting), sum(ff.*weighting), rtol=1e-12)
        RAPID2D.smooth_data_2D!(ff; num_SM, weighting)
        @test ff_SM == ff

        # test with zero weighting
        weighting[1, 1:3] .= 0.0
        weighting[end-3:end, end - 1] .= 0.0
        num_SM = 3
        ff_SM = RAPID2D.smooth_data_2D(ff; num_SM, weighting)
        @test ff_SM ≉ ff
        @test !isapprox(sum(ff_SM), sum(ff), rtol=1e-12)
        @test isapprox(sum(ff_SM.*weighting), sum(ff.*weighting), rtol=1e-12)
        RAPID2D.smooth_data_2D!(ff; num_SM, weighting)
        @test ff_SM == ff
    end
end
