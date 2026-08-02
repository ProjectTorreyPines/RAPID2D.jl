# The four real channels, expressed in the (v∥, λ∥, v⊥, λ⊥) basis.
#
# This models no new physics. Every channel already computes its four numbers and
# then destroys two by multiplying them into a single D; the adapters stop the
# return from being lossy. So the binding requirement is NOT that the adapters be
# plausible — it is that they reproduce, exactly, the D the solver already uses.

@testitem "Channel adapter: the H2 gas round-trips its own diffusivity" begin
    using RAPID2D: neutral_gas_channel, neutral_gas_diffusivity,
        neutral_gas_thermal_speed

    # The gas is one channel with three competing loss processes — elastic,
    # ionization, wall — all traversed at the SAME v_th, so they combine by
    # Matthiessen inside λ and the ceiling is counted once (§2.2).
    for (n, T, νiz, L) in (
            (6.4e17, 0.026, 0.0, 1.2),
            (1.0e15, 0.026, 1.0e4, 1.2),
            (1.0e19, 0.05, 1.0e6, 0.8),
            (0.0, 0.026, 0.0, 1.2),          # fully burnt: λ is the wall term alone
        )
        ch = neutral_gas_channel(n, T, νiz, L)
        # ½·v·λ must be the diffusivity the solver already computes — exactly
        @test 0.5 * ch.v_para * ch.λ_para ≈ neutral_gas_diffusivity(n, T, νiz, L) rtol = 1.0e-14
        # the speed is the D-convention one this module already defines
        @test ch.v_para ≈ neutral_gas_thermal_speed(T) rtol = 1.0e-14
        # a neutral gas has no preferred axis
        @test ch.v_perp == ch.v_para
        @test ch.λ_perp == ch.λ_para
    end
end

@testitem "Channel adapter: the turbulent ExB channel round-trips, 9:1 elongated" begin
    using RAPID2D: turbulent_ExB_channel

    # D_pol_turb = ½·v_ExB·L_mixing, split f∥ : f⊥ along b̂_pol
    E_pol, B_tot, L_mix = 37.5, 0.63, 1.0
    f_para, f_perp = 0.9, 0.1
    ch = turbulent_ExB_channel(E_pol, B_tot, L_mix, f_para, f_perp)

    v_ExB = E_pol / B_tot
    D_pol = 0.5 * v_ExB * L_mix
    @test ch.v_para ≈ v_ExB rtol = 1.0e-14
    @test ch.v_perp ≈ v_ExB rtol = 1.0e-14          # one speed, two step lengths
    @test 0.5 * ch.v_para * ch.λ_para ≈ D_pol * f_para rtol = 1.0e-14
    @test 0.5 * ch.v_perp * ch.λ_perp ≈ D_pol * f_perp rtol = 1.0e-14

    # the eddy is 9× longer along b̂_pol than across it — a field-aligned eddy,
    # which is what an ExB mixing model should produce
    @test ch.λ_para / ch.λ_perp ≈ f_para / f_perp rtol = 1.0e-14
    @test ch.λ_para / ch.λ_perp ≈ 9.0 rtol = 1.0e-14
end

@testitem "Channel adapter: the parallel collisional channel has no cross-field part" begin
    using RAPID2D: parallel_collisional_channel

    # D∥ = ½·v_p²/ν, so λ∥ = v_p/ν falls straight out
    v_p, ν = 1.5e6, 1.0e7
    D_para = 0.5 * v_p^2 / ν
    ch = parallel_collisional_channel(v_p, D_para)

    @test ch.v_para ≈ v_p rtol = 1.0e-14
    @test ch.λ_para ≈ v_p / ν rtol = 1.0e-14
    @test 0.5 * ch.v_para * ch.λ_para ≈ D_para rtol = 1.0e-14

    # streaming along B contributes nothing across it, so it must contribute
    # nothing at a wall the field points straight into (g = 1)
    @test ch.v_perp == 0.0
    @test 0.5 * ch.v_perp * ch.λ_perp == 0.0
end

@testitem "Channel adapter: the Bohm split is assumed, not derived" begin
    using RAPID2D: bohm_channel

    # D_B = Te/(16·B) is an empirical scaling, so it fixes only the PRODUCT
    # v⊥·λ⊥. We adopt λ⊥ = ρ_s, hence v⊥ = c_s/8 — a step of one sound gyroradius
    # per ≈1.3 gyro-periods. Design note §2.3.1.
    ee = 1.602176634e-19
    m_i = 1.673e-27
    Te, B = 5.0, 0.63
    ch = bohm_channel(Te, B, m_i)

    c_s = sqrt(Te * ee / m_i)
    ω_ci = ee * B / m_i
    ρ_s = c_s / ω_ci

    # the choice, pinned so it cannot drift silently
    @test ch.v_perp ≈ c_s / 8 rtol = 1.0e-12
    @test ch.λ_perp ≈ ρ_s rtol = 1.0e-12
    # and it must reproduce the empirical law exactly, by construction
    @test 0.5 * ch.v_perp * ch.λ_perp ≈ Te / (16 * B) rtol = 1.0e-12
    # a cross-field channel cannot reach a wall the field points into
    @test ch.v_para == 0.0

    # sanity on the adopted reading: ρ_s sub-millimetre, v⊥ a few km/s
    @test 1.0e-4 < ρ_s < 1.0e-3
    @test 2.0e3 < ch.v_perp < 4.0e3

    # THE MARKER. If Bohm diffusion and the turbulent ExB channel were the same
    # anomalous physics they would share a characteristic speed; they differ by
    # ~46×. So either they are distinct processes, one split is wrong, or the two
    # double-count the same transport. An UNEXPECTED PASS here means someone
    # unified them — reopen §2.3.1 rather than deleting this line.
    #
    # Stated as an explicit ratio window, not `isapprox`: `rtol = 1.0` reads as
    # "within a factor of two" but actually tests |x−y| ≤ max(|x|,|y|), which is
    # satisfied by almost any same-sign pair and would make the marker vacuous.
    v_ExB = 37.5 / 0.63
    @test_broken 0.5 < ch.v_perp / v_ExB < 2.0
    @test ch.v_perp / v_ExB > 40                  # records the present gap
end

@testitem "Channel adapter: Bohm and the collisional channel do not share a speed" begin
    using RAPID2D: bohm_channel, parallel_collisional_channel

    # §2.2's corollary: if a channel's D∥ and D⊥ do not yield the same v, it is
    # not one anisotropic channel but two — so their ceilings must be SUMMED
    # rather than combined by Matthiessen, even though the code assembles them
    # into one tensor.
    ee = 1.602176634e-19
    me, m_i = 9.109e-31, 1.673e-27
    Te, B = 5.0, 0.63

    v_p = sqrt(2 * Te * ee / me)
    coll = parallel_collisional_channel(v_p, 0.5 * v_p^2 / 1.0e7)
    bohm = bohm_channel(Te, B, m_i)

    @test coll.v_para / bohm.v_perp > 100         # measured ~12× at production Te…
    @test !(coll.v_para ≈ bohm.v_perp)            # …and never equal
end

@testitem "Channel adapter: summed channels reproduce the solver's own tensor" begin
    using RAPID2D: DiffusionChannel, diffusion_tensor, total_tensor,
        turbulent_ExB_channel, parallel_collisional_channel, bohm_channel

    # The integration test that makes the adapters binding. RAPID2D assembles
    # DRR/DRZ/DZZ from a base tensor aligned with the FULL field b̂ plus a
    # turbulent tensor aligned with the POLOIDAL field b̂_pol — two different axes.
    # Phase 1's total_tensor(channels, bR, bZ) assumed one shared axis, which is
    # not what the physics does; the per-channel-direction method is what makes
    # this reproducible.
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

    # base channel: whatever Dpara/Dperp the solver arrived at, expressed in the
    # basis — one speed is enough here because we only need the tensor back
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
    else
        DRR, DRZ, DZZ = base_T
    end

    # the basis reproduces what the solver assembled, to machine precision
    @test DRR ≈ tp.DRR rtol = 1.0e-12
    @test DRZ ≈ tp.DRZ rtol = 1.0e-12
    @test DZZ ≈ tp.DZZ rtol = 1.0e-12

    # and the per-channel-direction overload agrees with doing it by hand
    if RP.flags.turb_ExB_mixing
        fpara = RP.config.turbulent_diffusion_fraction_along_bpol
        fperp = 1 - fpara
        v_t = fill(1.0e3, NR, NZ)
        turb = DiffusionChannel(
            v_t, @.(2 * tp.Dpol_turb * fpara / v_t),
            v_t, @.(2 * tp.Dpol_turb * fperp / v_t)
        )
        sRR, sRZ, sZZ = total_tensor(
            ((base, F.bR, F.bZ), (turb, F.bpol_R, F.bpol_Z))
        )
        @test sRR ≈ tp.DRR rtol = 1.0e-12
        @test sRZ ≈ tp.DRZ rtol = 1.0e-12
        @test sZZ ≈ tp.DZZ rtol = 1.0e-12
    end
end
