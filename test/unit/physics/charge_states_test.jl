# `Zeff` was carrying three different quantities.
#
# In a single-species plasma the charge state Z, the mean charge Z̄ and the
# effective charge Z_eff are all equal to Z, so one field could stand in for all
# three and nothing complained. They separate the moment a second species exists:
#
#   Z_s   = the charge state of species s        → collision rates (NRL p.28/34)
#   Z̄     = Σ n_z Z_z / Σ n_z                    → quasineutrality, charge density
#   Z_eff = Σ n_z Z_z² / n_e                     → single-fluid closures (Spitzer)
#
# This branch resolves that by carrying ONE species, so all three collapse onto
# its declared charge and no derived field is needed to hold them apart. The
# multi-species forms — and why a stored average was the wrong shape — are in
# internal/docs/src/notes/TODO/ion-inventory-multi-species.md.

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

@testitem "The transport solve carries exactly one ion species" setup = [ChargeCase] begin
    using RAPID2D: IonSpecies, set_ion_species!, bulk_ion_charge

    RP = charge_case()
    @test length(RP.transport.ion_species) == 1
    @test bulk_ion_charge(RP) == 1              # H₂⁺

    # A second species is refused rather than half-supported. The wall pass clears
    # one column, γ_2nd is not per species, `Ni_loss` does not split, and the
    # charge density would need Σ n_z Z_z instead of n·Z — none of which exist yet.
    mi = RP.config.constants.mi
    err = try
        set_ion_species!(RP, [IonSpecies(:H2⁺, mi, 1), IonSpecies(:C⁶⁺, 6mi, 6)])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("one ion species", err.msg)
    @test occursin("wall", err.msg)              # names what is missing, not just "no"

    # …and the refusal leaves the existing species untouched
    @test length(RP.transport.ion_species) == 1
    @test bulk_ion_charge(RP) == 1
end

@testitem "There is no stored mean charge — Z comes from the species" setup = [ChargeCase] begin
    using RAPID2D: IonSpecies, set_ion_species!, bulk_ion_charge

    RP = charge_case()

    # `Z_mean` existed only to let `ni` mean "total ion density". With one species
    # `ni` IS that species, so a per-ion average has nothing to average over.
    @test !hasproperty(RP.plasma, :Z_mean)

    # `Z_eff` survives — `sptz_fac` is a genuine consumer — and at one species it
    # is that species' charge exactly, not a value maintained alongside it.
    mi = RP.config.constants.mi
    set_ion_species!(RP, [IonSpecies(:C⁶⁺, 6mi, 6)])
    update_charge_states!(RP)
    @test bulk_ion_charge(RP) == 6
    @test all(==(6.0), RP.plasma.Zeff)
end

@testitem "Quasineutrality slaves ions with the declared species charge" setup = [ChargeCase] begin
    using RAPID2D: IonSpecies, set_ion_species!, slave_ions_to_electrons!

    # n_e = n_i Z, so the ion density balancing a given n_e is ne/Z. Reading Z from
    # the declared species rather than from a field means it CANNOT be stale: the
    # test below never refreshes any derived quantity between declaring C⁶⁺ and
    # slaving, which is exactly the window a stored average got wrong.
    RP = charge_case()
    mi = RP.config.constants.mi
    set_ion_species!(RP, [IonSpecies(:C⁶⁺, 6mi, 6)])
    slave_ions_to_electrons!(RP)
    @test RP.plasma.ni ≈ RP.plasma.ne ./ 6

    # With one singly-charged species that is exactly ni == ne, bitwise
    RP2 = charge_case()
    RP2.flags.update_ni_independently = false
    run_simulation!(RP2)
    @test RP2.plasma.ni == RP2.plasma.ne
end

@testitem "A current carries the ion CHARGE density, not the ion density" setup = [ChargeCase] begin
    using RAPID2D: IonSpecies, set_ion_species!, measure_snap2D

    # J∥ = e(n_i Z u_i∥ − n_e u_e∥). The ion term is a charge density; dropping the
    # Z under-reports the ion current by a factor of Z, silently, because H₂⁺ makes
    # Z = 1 and the omission invisible. Read from the species, so declaring a
    # different ion is enough — no derived field has to be refreshed in between.
    RP = charge_case()
    mi = RP.config.constants.mi
    RP.plasma.ue_para .= -1.0e5
    RP.plasma.ui_para .= 2.0e3
    ee = RP.config.constants.ee

    set_ion_species!(RP, [IonSpecies(:H2⁺, mi, 1)])
    J_H2 = copy(measure_snap2D(RP).J_para)

    set_ion_species!(RP, [IonSpecies(:C⁶⁺, 6mi, 6)])
    J_C = measure_snap2D(RP).J_para

    pla = RP.plasma
    @test J_H2 ≈ @. ee * (pla.ni * 1 * pla.ui_para - pla.ne * pla.ue_para)
    @test J_C ≈ @. ee * (pla.ni * 6 * pla.ui_para - pla.ne * pla.ue_para)

    # …and the difference is exactly the extra five charges per ion
    @test J_C - J_H2 ≈ @. ee * 5 * pla.ni * pla.ui_para
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

@testitem "The new collision field reaches the 2D snapshot" setup = [ChargeCase] begin
    using RAPID2D: measure_snap2D

    # `ν_ii` now carries its own logarithm, and that correction moves the ion AND
    # (through the ambipolar coupling) the ELECTRON parallel diffusivity by tens
    # of percent. A user whose run moves has to be able to see why, so the new
    # field has to be observable, not just internal.
    RP = charge_case()
    update_transport_quantities!(RP)
    snap = measure_snap2D(RP)

    @test snap.lnΛ_ii == RP.plasma.lnΛ_ii
    @test snap.lnΛ_ii != snap.lnΛ                 # Te ≠ Ti ⇒ genuinely different
    @test all(>(0.0), snap.lnΛ_ii)

    # A charge AVERAGE is not a 2D field here: one species means one scalar charge,
    # and recording NR×NZ copies of it every snapshot says nothing.
    @test !hasproperty(snap, :Z_mean)
end
