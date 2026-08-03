# The four real channels, expressed in the (v∥, λ∥, v⊥, λ⊥) basis.
#
# This models no new physics. Every channel already computes its four numbers and
# then destroys two by multiplying them into a single D; the adapters stop the
# return from being lossy. So the binding requirement is NOT that the adapters be
# plausible — it is that they reproduce, exactly, the D the solver already uses,
# on the FIELDS the solver actually holds.

@testitem "Channel adapter: the H2 gas round-trips its own diffusivity" begin
    using RAPID2D: neutral_gas_channel, neutral_gas_diffusivity,
        neutral_gas_thermal_speed, DiffusionChannel

    # The gas is one channel with three competing loss processes — elastic,
    # ionization, wall — all traversed at the SAME v_th, so they combine by
    # Matthiessen inside λ and the ceiling is counted once.
    for (n, T, νiz, L) in (
            (6.4e17, 0.026, 0.0, 1.2),
            (1.0e15, 0.026, 1.0e4, 1.2),
            (1.0e19, 0.05, 1.0e6, 0.8),
            (0.0, 0.026, 0.0, 1.2),          # fully burnt: λ is the wall term alone
        )
        ch = neutral_gas_channel(fill(n, 1, 1), T, νiz, L)
        @test ch isa DiffusionChannel{Float64}
        # ½·v·λ must be the diffusivity the solver already computes — exactly
        @test only(@. 0.5 * ch.v_para * ch.λ_para) ≈
            neutral_gas_diffusivity(n, T, νiz, L) rtol = 1.0e-14
        # the speed is the D-convention one this module already defines
        @test only(ch.v_para) ≈ neutral_gas_thermal_speed(T) rtol = 1.0e-14
        # a neutral gas has no preferred axis
        @test ch.v_perp == ch.v_para
        @test ch.λ_perp == ch.λ_para
    end
end

@testitem "Channel adapter: the turbulent ExB channel round-trips, 9:1 elongated" begin
    using RAPID2D: turbulent_ExB_channel, DiffusionChannel

    # D_pol_turb = ½·v_ExB·L_mixing, split f∥ : f⊥ along b̂_pol
    E_pol, B_tot, L_mix = 37.5, 0.63, 1.0
    f_para, f_perp = 0.9, 0.1
    ch = turbulent_ExB_channel(fill(E_pol, 1, 1), fill(B_tot, 1, 1), L_mix, f_para, f_perp)
    @test ch isa DiffusionChannel{Float64}

    v_ExB = E_pol / B_tot
    D_pol = 0.5 * v_ExB * L_mix
    @test only(ch.v_para) ≈ v_ExB rtol = 1.0e-14
    @test only(ch.v_perp) ≈ v_ExB rtol = 1.0e-14          # one speed, two step lengths
    @test only(@. 0.5 * ch.v_para * ch.λ_para) ≈ D_pol * f_para rtol = 1.0e-14
    @test only(@. 0.5 * ch.v_perp * ch.λ_perp) ≈ D_pol * f_perp rtol = 1.0e-14

    # the eddy is 9× longer along b̂_pol than across it — a field-aligned eddy,
    # which is what an ExB mixing model should produce
    @test only(@. ch.λ_para / ch.λ_perp) ≈ f_para / f_perp rtol = 1.0e-14
    @test only(@. ch.λ_para / ch.λ_perp) ≈ 9.0 rtol = 1.0e-14
end

@testitem "Channel adapter: the parallel collisional channel has no cross-field part" begin
    using RAPID2D: parallel_collisional_channel

    # D∥ = ½·v_p²/ν, so λ∥ = v_p/ν falls straight out
    v_p, ν = 1.5e6, 1.0e7
    D_para = 0.5 * v_p^2 / ν
    ch = parallel_collisional_channel(fill(v_p, 1, 1), fill(D_para, 1, 1))

    @test only(ch.v_para) ≈ v_p rtol = 1.0e-14
    @test only(ch.λ_para) ≈ v_p / ν rtol = 1.0e-14
    @test only(@. 0.5 * ch.v_para * ch.λ_para) ≈ D_para rtol = 1.0e-14

    # streaming along B contributes nothing across it, so it must contribute
    # nothing at a wall the field points straight into
    @test all(iszero, ch.v_perp)
    @test all(iszero, @. 0.5 * ch.v_perp * ch.λ_perp)
end

@testitem "Channel adapter: the Bohm split is assumed, not derived" begin
    using RAPID2D: bohm_channel

    # D_B = Te/(16·B) is an empirical scaling, so it fixes only the PRODUCT
    # v⊥·λ⊥. We adopt λ⊥ = ρ_s, hence v⊥ = c_s/8 — a step of one sound gyroradius
    # per ≈1.3 gyro-periods.
    ee = 1.602176634e-19
    m_i = 1.673e-27
    Te, B = 5.0, 0.63
    ch = bohm_channel(fill(Te, 1, 1), fill(B, 1, 1), m_i)

    c_s = sqrt(Te * ee / m_i)
    ω_ci = ee * B / m_i
    ρ_s = c_s / ω_ci

    # the choice, pinned so it cannot drift silently
    @test only(ch.v_perp) ≈ c_s / 8 rtol = 1.0e-12
    @test only(ch.λ_perp) ≈ ρ_s rtol = 1.0e-12
    # and it must reproduce the empirical law exactly, by construction
    @test only(@. 0.5 * ch.v_perp * ch.λ_perp) ≈ Te / (16 * B) rtol = 1.0e-12
    # a cross-field channel cannot reach a wall the field points into
    @test all(iszero, ch.v_para)

    # sanity on the adopted reading: ρ_s sub-millimetre, v⊥ a few km/s
    @test 1.0e-4 < ρ_s < 1.0e-3
    @test 2.0e3 < only(ch.v_perp) < 4.0e3

    # THE MARKER. If Bohm diffusion and the turbulent ExB channel were the same
    # anomalous physics they would share a characteristic speed; they differ by
    # ~46×. So either they are distinct processes, one split is wrong, or the two
    # double-count the same transport. An UNEXPECTED PASS here means someone
    # unified them — reopen the assumption rather than deleting this line.
    #
    # Stated as an explicit ratio window, not `isapprox`: `rtol = 1.0` reads as
    # "within a factor of two" but actually tests |x−y| ≤ max(|x|,|y|), which is
    # satisfied by almost any same-sign pair and would make the marker vacuous.
    v_ExB = 37.5 / 0.63
    @test_broken 0.5 < only(ch.v_perp) / v_ExB < 2.0
    @test only(ch.v_perp) / v_ExB > 40                  # records the present gap
end

@testitem "Channel adapter: Bohm and the collisional channel do not share a speed" begin
    using RAPID2D: bohm_channel, parallel_collisional_channel

    # If a channel's D∥ and D⊥ do not yield the same v it is not one anisotropic
    # channel but two — so their ceilings must be SUMMED rather than combined by
    # Matthiessen, even though the code assembles them into one tensor.
    ee = 1.602176634e-19
    me, m_i = 9.109e-31, 1.673e-27
    Te, B = 5.0, 0.63

    v_p = sqrt(2 * Te * ee / me)
    coll = parallel_collisional_channel(fill(v_p, 1, 1), fill(0.5 * v_p^2 / 1.0e7, 1, 1))
    bohm = bohm_channel(fill(Te, 1, 1), fill(B, 1, 1), m_i)

    @test only(coll.v_para) / only(bohm.v_perp) > 100
    @test !(only(coll.v_para) ≈ only(bohm.v_perp))
end

@testitem "Channel adapter: field inputs stay elementwise on a non-square grid" begin
    using RAPID2D: turbulent_ExB_channel, parallel_collisional_channel, bohm_channel,
        neutral_gas_channel, diffusion_tensor, total_tensor, DiffusionChannel

    # These take FIELDS, and `/` between two matrices is right-division rather
    # than division: on a square grid it throws SingularException for a constant
    # field, and on a non-square one it silently returns the wrong shape. NR ≠ NZ
    # on purpose — a square grid would hide the second failure mode entirely.
    NR, NZ = 4, 7
    E = fill(37.5, NR, NZ)
    B = fill(0.63, NR, NZ)
    Te = fill(5.0, NR, NZ)
    m_i = 1.673e-27

    channels = (
        turbulent_ExB_channel(E, B, 1.0, 0.9, 0.1),
        parallel_collisional_channel(fill(1.5e6, NR, NZ), fill(1.125e5, NR, NZ)),
        bohm_channel(Te, B, m_i),
        neutral_gas_channel(fill(6.4e17, NR, NZ), 0.026, 0.0, 1.2),
    )
    for ch in channels
        @test ch isa DiffusionChannel{Float64}
        @test size(ch.v_para) == (NR, NZ)
        @test size(ch.λ_para) == (NR, NZ)
        @test size(ch.v_perp) == (NR, NZ)
        @test size(ch.λ_perp) == (NR, NZ)
        # elementwise means a uniform input gives the same value at every node
        @test all(≈(first(ch.v_para)), ch.v_para)
        @test all(isfinite, ch.λ_para)
        @test all(isfinite, ch.λ_perp)
    end

    # …and the results flow straight into the advertised consumers, which is the
    # whole point of returning a channel rather than a bag of numbers
    bR, bZ = fill(0.6, NR, NZ), fill(0.8, NR, NZ)
    for ch in channels
        D_RR, D_RZ, D_ZZ = diffusion_tensor(ch, bR, bZ)
        @test size(D_RR) == (NR, NZ)
        @test all(≥(0), D_RR)
        @test all(≥(0), D_ZZ)
    end
    tRR, _, _ = total_tensor(channels, bR, bZ)
    @test size(tRR) == (NR, NZ)
    # fluxes add, so tensors add
    @test tRR ≈ sum(diffusion_tensor(ch, bR, bZ)[1] for ch in channels) rtol = 1.0e-14
end

@testitem "Channel adapter: a scalar-only call is rejected, not silently reshaped" begin
    using RAPID2D: DiffusionChannel

    # A channel lives on the grid. Accepting an all-scalar call would build a
    # 0-dimensional channel that fails much later, somewhere less obvious.
    @test_throws ArgumentError DiffusionChannel(1.0, 2.0, 3.0, 4.0)
    # a genuine shape clash is Julia's own error, not a silent reshape
    @test_throws DimensionMismatch DiffusionChannel(
        zeros(2, 3), zeros(4, 5), zeros(2, 3), zeros(2, 3)
    )
    # mixing one field with scalars is the normal case and must work
    ch = DiffusionChannel(fill(1.0, 2, 3), 2.0, 3.0, 4.0)
    @test size(ch.v_para) == (2, 3)
    @test all(==(2.0), ch.λ_para)
end

@testitem "Channel adapter: summed channels reproduce the solver's own tensor" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor, total_tensor

    # The integration test that makes the adapters binding. RAPID2D assembles
    # DRR/DRZ/DZZ from a base tensor aligned with the FULL field b̂ plus a
    # turbulent tensor aligned with the POLOIDAL field b̂_pol — two different axes,
    # which is why the per-channel-direction method exists.
    config = SimulationConfig{Float64}(
        device_Name = "manual", NR = 15, NZ = 17,
        prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-6,
        snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
    )
    RP = RAPID{Float64}(config)
    initialize!(RP)
    RP.plasma.ne .= 1.0e18
    RP.plasma.Te_eV .= 5.0
    RAPID2D.update_transport_quantities!(RP)

    G, tp, F = RP.G, RP.transport, RP.fields
    NR, NZ = G.NR, G.NZ

    v_ref = fill(1.0e6, NR, NZ)
    base = DiffusionChannel(
        v_ref, @.(2 * tp.Dpara / v_ref),
        v_ref, @.(2 * tp.Dperp / v_ref)
    )
    base_T = diffusion_tensor(base, F.bR, F.bZ)

    if RP.flags.turb_ExB_mixing
        fpara = RP.config.turbulent_diffusion_fraction_along_bpol
        fperp = 1 - fpara
        v_t = fill(1.0e3, NR, NZ)
        turb = DiffusionChannel(
            v_t, @.(2 * tp.Dpol_turb * fpara / v_t),
            v_t, @.(2 * tp.Dpol_turb * fperp / v_t)
        )
        turb_T = diffusion_tensor(turb, F.bpol_R, F.bpol_Z)
        DRR = base_T[1] .+ turb_T[1]
        DRZ = base_T[2] .+ turb_T[2]
        DZZ = base_T[3] .+ turb_T[3]

        # and the per-channel-direction overload agrees with doing it by hand
        sRR, sRZ, sZZ = total_tensor(
            ((base, F.bR, F.bZ), (turb, F.bpol_R, F.bpol_Z))
        )
        @test sRR ≈ tp.DRR rtol = 1.0e-12
        @test sRZ ≈ tp.DRZ rtol = 1.0e-12
        @test sZZ ≈ tp.DZZ rtol = 1.0e-12
    else
        DRR, DRZ, DZZ = base_T
    end

    # the basis reproduces what the solver assembled, to machine precision
    @test DRR ≈ tp.DRR rtol = 1.0e-12
    @test DRZ ≈ tp.DRZ rtol = 1.0e-12
    @test DZZ ≈ tp.DZZ rtol = 1.0e-12
end
