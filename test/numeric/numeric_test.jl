using Test
using RAPID2D

@testset "Gradient Calculation Tests" begin

    FT = Float64
    # Create simulation configuration
    config = SimulationConfig{FT}(
        NR=100, NZ=200,
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