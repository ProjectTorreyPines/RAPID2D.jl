# The current column's 0D force balance against a uniform circular current disc: its geometry,
# l_i = 1/2, the hoop and tire-tube forces, the Shafranov field it needs and where it then sits,
# and the decay index of a curved external field.

@testsnippet CurrentDisc begin
    using RAPID2D: column_force_balance, solve_Ampere_equation!
    const μ0 = 4π * 1.0e-7
    const R0, Z0 = 1.6, 0.1             # on the nodes of the 81 × 121 grid (ΔR = ΔZ = 0.02 m)

    "A uniform disc of current `I` (radius 0.205 m, no node on its edge) at (R0, Z0), with
    nₑ = nᵢ = `n` and Tₑ = Tᵢ = `T` on it, in the uniform external B_Z = `B0`; the self field
    solved from the disc."
    function current_disc(; I = -1.0e4, B0 = 0.0, n = 1.0e19, T = 10.0)
        config = SimulationConfig{Float64}(;
            device_Name = "manual", manual = ManualSetup{Float64}(BR = 0.0, BZ = B0),
            NR = 81, NZ = 121, R0B0 = 3.0, prefilled_gas_pressure = 0.0, dt = 1.0e-6,
            snap0D_Δt_s = 1.0e-6, snap2D_Δt_s = 1.0e-6, Output_path = mktempdir(),
        )
        RP = RAPID{Float64}(config)
        RP.flags.Ampere = true                  # builds the boundary Green function the self field needs
        initialize!(RP)
        G, pla = RP.G, RP.plasma
        disc = [k for k in G.nodes.in_wall_nids if (G.R2D[k] - R0)^2 + (G.Z2D[k] - Z0)^2 <= 0.205^2]
        foreach(fill!, (pla.Jϕ, pla.ne, pla.ni), 0.0)
        pla.Jϕ[disc] .= I / (length(disc) * G.dR * G.dZ)
        pla.ne[disc] .= n
        pla.ni[disc] .= n
        fill!(pla.Te_eV, T)
        fill!(pla.Ti_eV, T)
        solve_Ampere_equation!(RP; update_Eϕ_self = false)
        area = length(disc) * G.dR * G.dZ
        return RP, (; I, area, a = sqrt(area / π), p0 = RP.config.constants.ee * n * 2T)
    end
end

@testitem "column force balance: a uniform disc's geometry, l_i, β_p, hoop and tire-tube forces" setup = [CurrentDisc] begin
    RP, d = current_disc()
    fb = column_force_balance(RP)
    @test fb.R_centroid ≈ R0 atol = 1.0e-9
    @test fb.Z_centroid ≈ Z0 atol = 1.0e-9
    @test fb.a_eff ≈ d.a rtol = 0.03
    @test fb.li ≈ 0.5 rtol = 0.05                                  # uniform current (0.519 here, 0.498 at twice the resolution)
    @test fb.beta_p ≈ 4 * d.p0 * 2π * R0 * d.area / (μ0 * d.I^2 * R0) rtol = 0.05
    # the self field's radial force is the hoop force of a ring, (μ0 I²/2)(ln(8R/a) − 1 + l_i/2)
    hoop = μ0 * d.I^2 / 2 * (log(8R0 / d.a) - 1 + fb.li / 2)
    @test fb.force_R_self ≈ hoop rtol = 0.02
    @test abs(fb.force_Z_self) < 1.0e-2 * abs(fb.force_R_self)     # an isolated ring: no vertical self-force
    @test fb.force_R_pressure ≈ 2π * d.p0 * d.area rtol = 1.0e-10
    @test fb.force_R_toroidal ≈ μ0 * d.I^2 / 4 * (fb.beta_p - 1)
    # no external field: both estimates of the vertical field to add agree
    @test fb.force_R_ext == 0 && fb.BZ_ext_centroid == 0
    @test fb.Bv_to_add ≈ fb.Bv_req rtol = 0.02
end

@testitem "column force balance: in the vertical field it needs, the column sits at its centroid" setup = [CurrentDisc] begin
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, d = current_disc(; B0)
    fb = column_force_balance(RP)
    Λ = -4π * R0 * B0 / (μ0 * d.I)
    @test sign(B0) == -sign(d.I)                                   # it pushes the column inward
    @test fb.Bv_req ≈ B0 rtol = 1.0e-12
    @test fb.BZ_ext_centroid ≈ B0 rtol = 1.0e-12
    @test fb.decay_index == 0                                      # a uniform field
    @test fb.decay_index_radial_limit ≈ 1 - 1 / Λ
    @test fb.force_R_ext ≈ 2π * B0 * d.I * R0 rtol = 1.0e-10
    @test fb.force_Z_ext == 0
    @test fb.R_eq ≈ R0 atol = 1.0e-3
    @test fb.R_eq_stiffness ≈ μ0 * d.I^2 / (2R0) * (1 - Λ) rtol = 0.02   # dF_R/dR at n = 0
    @test fb.R_eq_stiffness < 0                                    # restoring
    @test abs(fb.Bv_to_add) < 0.02 * abs(B0)                       # the model's own forces agree
end

@testitem "column force balance: the decay index of a curved field, and its vertical restoring force" setup = [CurrentDisc] begin
    RP0, _ = current_disc()
    B0 = column_force_balance(RP0).Bv_req
    RP, d = current_disc(; B0)
    G, F = RP.G, RP.fields
    n_decay = 0.5                                                   # MATLAB's curved control field
    @. F.BZ_ext = B0 * (R0 / G.R2D)^n_decay
    @. F.BR_ext = -n_decay * B0 * (R0 / G.R2D)^n_decay * G.Z2D / G.R2D   # curl-free to first order in Z
    fb = column_force_balance(RP)
    @test fb.decay_index ≈ n_decay rtol = 0.02
    @test fb.force_Z_ext * Z0 < 0                                  # above the midplane, pushed back down
end

@testitem "column force balance: nothing below the current threshold; the snapshot carries it" setup = [CurrentDisc] begin
    using RAPID2D: measure_snap0D!, COLUMN_FORCE_BALANCE_FIELDS
    RP, _ = current_disc()
    snap = Snapshot0D{Float64}()
    measure_snap0D!(RP, snap)
    fb = column_force_balance(RP)
    @test all(getfield(snap, name) == fb[name] for name in COLUMN_FORCE_BALANCE_FIELDS)
    @test snap.Bv_req != 0
    fill!(RP.plasma.Jϕ, 0.0)
    @test all(iszero, column_force_balance(RP))
end
