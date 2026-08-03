# `Zeff` was carrying three different quantities.
#
# In a single-species hydrogen plasma the charge state Z, the mean charge Z̄ and
# the effective charge Z_eff are all 1, so one field could stand in for all three
# and nothing complained. They separate the moment a second species exists:
#
#   Z_s   = the charge state of species s        → collision rates (NRL p.28/34)
#   Z̄     = Σ n_z Z_z / Σ n_z                    → quasineutrality, charge density
#   Z_eff = Σ n_z Z_z² / n_e                     → single-fluid closures (Spitzer)
#
# For H₂⁺ + C⁶⁺ at 99:1 these are 1, 1.05 and 1.29; at 90:10, 1, 1.5 and 3.0.
# Fixing this AFTER adding impurities would mean re-deriving which of the three
# every existing formula meant, from formulas that all read `Zeff`.

@testsnippet ChargeCase begin
    "A discharge-shaped setup, ions free to evolve."
    function charge_case(; NR = 9, NZ = 9, kw...)
        config = SimulationConfig{Float64}(
            device_Name = "manual", NR = NR, NZ = NZ,
            R_min = 1.0, R_max = 2.0, Z_min = -0.5, Z_max = 0.5,
            wall_R = [1.15, 1.85, 1.85, 1.15], wall_Z = [-0.35, -0.35, 0.35, 0.35],
            prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0, dt = 1.0e-8,
            t_end_s = 1.0e-6, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0;
            kw...
        )
        RP = RAPID{Float64}(config)
        initialize!(RP)
        RP.plasma.ne .= 1.0e18
        RP.plasma.ni .= 1.0e18
        RP.plasma.Te_eV .= 10.0
        RP.plasma.Ti_eV .= 1.0
        return RP
    end
end

@testitem "Mean charge and effective charge are different averages" begin
    using RAPID2D: mean_charge, effective_charge

    # One species: all three collapse onto its charge state, whatever that is
    for Z in (1.0, 6.0)
        @test mean_charge([Z], [1.0e18]) ≈ Z
        @test effective_charge([Z], [1.0e18]) ≈ Z
    end

    # H₂⁺ 99 : C⁶⁺ 1 — Z̄ is a per-ION average, Z_eff a per-ELECTRON one
    Z, n = [1.0, 6.0], [9.9e17, 1.0e16]
    ne = n[1] * 1 + n[2] * 6
    @test mean_charge(Z, n) ≈ (n[1] * 1 + n[2] * 6) / (n[1] + n[2])
    @test effective_charge(Z, n) ≈ (n[1] * 1 + n[2] * 36) / ne
    @test mean_charge(Z, n) ≈ 1.05 rtol = 0.01
    @test effective_charge(Z, n) ≈ 1.286 rtol = 0.01    # 1 % carbon by ION number

    # Z_eff ≥ Z̄ always: squaring weights the high-Z tail harder. The gap widens
    # fast — at 10 % carbon it is 1.5 against 3.0, exactly a factor of two.
    @test effective_charge(Z, n) > mean_charge(Z, n)
    @test mean_charge(Z, [9.0e17, 1.0e17]) ≈ 1.5 rtol = 0.01
    @test effective_charge(Z, [9.0e17, 1.0e17]) ≈ 3.0 rtol = 0.01
end

@testitem "Charge averages survive an empty or negative plasma" begin
    using RAPID2D: mean_charge, effective_charge

    # A continuity solve is free to hand back n ≤ 0, and out-of-wall nodes are 0.
    # 0/0 in a charge would propagate into ν_ii, the current density and the
    # quasineutrality slaving all at once.
    @test mean_charge([1.0, 6.0], [0.0, 0.0]) == 1.0        # falls back to the bulk
    @test effective_charge([1.0, 6.0], [0.0, 0.0]) == 1.0
    @test isfinite(mean_charge([2.0], [-1.0e-40]))
    @test mean_charge([2.0], [-1.0e-40]) == 2.0
end

@testitem "update_charge_states! fills both fields from the species list" setup = [ChargeCase] begin
    using RAPID2D: IonSpecies, set_ion_species!, mean_charge, effective_charge

    RP = charge_case()
    update_charge_states!(RP)
    @test all(==(1.0), RP.plasma.Z_mean)
    @test all(==(1.0), RP.plasma.Zeff)

    # Add carbon and the two fields separate
    mi = RP.config.constants.mi
    set_ion_species!(
        RP, [
            IonSpecies(:H2⁺, mi, 1),
            IonSpecies(:C⁶⁺, 6mi, 6),
        ]
    )
    RP.transport.ion_N[:, 2] .= 1.0e16
    update_charge_states!(RP)

    Z, n = [1.0, 6.0], [1.0e18, 1.0e16]
    @test all(≈(mean_charge(Z, n)), RP.plasma.Z_mean)
    @test all(≈(effective_charge(Z, n)), RP.plasma.Zeff)
    @test RP.plasma.Zeff[5, 5] > RP.plasma.Z_mean[5, 5]

    # column 1 is `ni` by definition, so the sync must be exact
    @test RP.transport.ion_N[:, 1] == vec(RP.plasma.ni)
end

@testitem "Collision rates follow the species charge, not Zeff" setup = [ChargeCase] begin
    # The NRL rates on p.28 carry the CHARGE STATE of the colliding ion. Reading
    # them from `Zeff` was only ever right because the two coincided.
    RP = charge_case()
    update_coulomb_collision_parameters!(RP)
    ν_ii, ν_ei, lnΛ = copy(RP.plasma.ν_ii), copy(RP.plasma.ν_ei), copy(RP.plasma.lnΛ)

    # A hand-set Z_eff is a statement about the single-fluid closure, not about
    # what the bulk ion is. It must not reach a collision rate.
    RP.plasma.Zeff .= 4.0
    update_coulomb_collision_parameters!(RP)
    @test RP.plasma.ν_ii ≈ ν_ii rtol = 1.0e-14
    @test RP.plasma.ν_ei ≈ ν_ei rtol = 1.0e-14
    @test RP.plasma.lnΛ ≈ lnΛ rtol = 1.0e-14

    # …but the Spitzer factor is a genuine Z_eff consumer and MUST follow it
    @test !all(≈(0.510469472194728), RP.plasma.sptz_fac)
end

@testitem "The new collision and charge fields reach the 2D snapshot" setup = [ChargeCase] begin
    using RAPID2D: measure_snap2D

    # `ν_ii` now carries its own logarithm, and that correction moves the ion AND
    # (through the ambipolar coupling) the ELECTRON parallel diffusivity by tens
    # of percent. A user whose run moves has to be able to see why, so the two
    # new fields have to be observable, not just internal.
    RP = charge_case()
    update_transport_quantities!(RP)
    snap = measure_snap2D(RP)

    @test snap.lnΛ_ii == RP.plasma.lnΛ_ii
    @test snap.Z_mean == RP.plasma.Z_mean
    @test snap.lnΛ_ii != snap.lnΛ                 # Te ≠ Ti ⇒ genuinely different
    @test all(>(0.0), snap.lnΛ_ii)
end

@testitem "Quasineutrality slaves ions with the mean charge" setup = [ChargeCase] begin
    using RAPID2D: slave_ions_to_electrons!

    # n_e = Σ n_z Z_z = n_i Z̄, so the ion density that balances a given n_e is
    # ne/Z̄ — NOT ne/Z_eff, and not ne except when Z̄ = 1.
    RP = charge_case()
    RP.plasma.Z_mean .= 1.5
    RP.plasma.Zeff .= 4.0
    slave_ions_to_electrons!(RP)
    @test RP.plasma.ni ≈ RP.plasma.ne ./ 1.5

    # The two places that slave ions now agree by construction: both call this.
    RP2 = charge_case()
    RP2.flags.update_ni_independently = false
    run_simulation!(RP2)
    @test RP2.plasma.ni ≈ RP2.plasma.ne ./ RP2.plasma.Z_mean

    # and with one hydrogen species that is exactly ni == ne, so nothing moves
    @test all(==(1.0), RP2.plasma.Z_mean)
    @test RP2.plasma.ni == RP2.plasma.ne
end
