# Closure identities on the RRC table itself. No time integration, no RAPID object:
# a failure here localizes to the data or the reader, never to the equations.
@testitem "RRC ledger: the installed table carries the 2026-08 group schema" begin
    using RAPID2D.HDF5

    path = joinpath(pkgdir(RAPID2D), "RRC_data", "eRRCs_EoverP_Erg.h5")
    names = h5open(keys, path)

    # The particle, momentum and energy ledgers, by BD's dataset names. RAPID2D renames
    # `L_*` to `Kerg_*` on read (Task B1); the FILE still spells them `L_*`.
    surfaces = (
        "K_iz", "K_diss_iz", "K_exc", "K_diss_exc", "K_ela",
        "K_mom", "K_mom_by_ela", "K_mom_by_exc", "K_mom_by_diss_exc",
        "K_mom_by_iz", "K_mom_by_diss_iz",
        "L_ela", "L_exc", "L_diss_exc", "L_tot",
    )
    for k in surfaces
        @test k in names
    end

    # Presence alone does not catch an inverted axis order: BD's (T, u_d) ion tables
    # were once stored transposed, and "nothing detects the mismatch: it silently
    # returns the right number at the wrong point" — see
    # internal/docs/src/notes/issues/ion-rrc-table-transposed.md (fixed in d3a5343).
    # Every (E/p, Ē) surface here must be `(length(EoverP), length(Erg_eV))`.
    EoverP, Erg_eV = h5open(path) do f
        read(f, "EoverP"), read(f, "Erg_eV")
    end
    expected_size = (length(EoverP), length(Erg_eV))
    h5open(path) do f
        for k in surfaces
            @test size(read(f, k)) == expected_size
        end
    end
end

@testitem "RRC ledger: the energy parts sum to L_tot" begin
    using RAPID2D.HDF5

    path = joinpath(pkgdir(RAPID2D), "RRC_data", "eRRCs_EoverP_Erg.h5")
    d = h5open(path) do f
        Dict(k => read(f, k) for k in ("L_ela", "L_exc", "L_diss_exc", "L_tot", "K_iz", "K_diss_iz"))
    end

    # BD assembles L_tot from the exported parts AFTER hard-zeroing, so this closes by
    # construction. A mismatch means the assembly here is wrong, not the table.
    #   L_tot = L_ela + L_exc + L_diss_exc + e*(15.426*K_iz + 35.0*K_diss_iz)
    e_C = 1.602176634e-19
    assembled = @. d["L_ela"] + d["L_exc"] + d["L_diss_exc"] +
        e_C * (15.426 * d["K_iz"] + 35.0 * d["K_diss_iz"])

    tot = d["L_tot"]
    live = tot .> 0.0
    @test any(live)                                   # the surface is not all zeros
    @test maximum(abs.(assembled[live] .- tot[live]) ./ tot[live]) < 1.0e-10
    # Dead cells must be dead on BOTH sides: a part that survives where the total is
    # zeroed would mean the masks disagree.
    @test all(assembled[.!live] .== 0.0)
end

@testitem "RRC ledger: the momentum shares sum to K_mom" begin
    using RAPID2D.HDF5
    using RAPID2D.Statistics

    path = joinpath(pkgdir(RAPID2D), "RRC_data", "eRRCs_EoverP_Erg.h5")
    d = h5open(path) do f
        Dict(
            k => read(f, k) for k in (
                    "K_mom", "K_mom_by_ela", "K_mom_by_exc",
                    "K_mom_by_diss_exc", "K_mom_by_iz", "K_mom_by_diss_iz",
                )
        )
    end

    assembled = @. d["K_mom_by_ela"] + d["K_mom_by_exc"] + d["K_mom_by_diss_exc"] +
        d["K_mom_by_iz"] + d["K_mom_by_diss_iz"]

    tot = d["K_mom"]
    live = tot .> 0.0
    @test any(live)

    # Independent reconstructions, NOT exact — unlike the energy ledger, which BD
    # assembles from its own parts and which closes to 1e-10. K_mom is a separately
    # accumulated v_z-weighted moment, and make_envelope_h5.jl applies floors and the
    # cold-band substitution per surface, so the sum of the parts drifts from the
    # total. Measured on the 2026-08 table: median 4.5e-5, max 4.8e-3.
    #
    # The bounds are set from that measurement with margin, and they still bite: the
    # SMALLEST share, K_mom_by_diss_iz, is ~7 % of K_mom at high E/p, so a channel
    # bound to the wrong dataset or dropped from the sum lands one to three orders
    # above this residual.
    rel = abs.(assembled[live] .- tot[live]) ./ tot[live]
    @test median(rel) < 1.0e-3
    @test maximum(rel) < 2.0e-2
end

@testitem "RRC ledger: the installed table carries no legacy surface" begin
    using RAPID2D.HDF5
    # The compile-time proof that nothing reads a pre-2026-08 name. A grep cannot see a
    # name built at runtime; a missing dataset can only fail loudly.
    #
    # This replaces the testitem that pinned the cold-band defect in `Total_Excitation`
    # (the alias carried the raw EXC-group rate below 0.0388 eV, billing rotational
    # excitation at 12 eV instead of 0.0441 eV). That surface no longer ships, so the
    # defect is retired rather than merely unread.
    path = joinpath(pkgdir(RAPID2D), "RRC_data", "eRRCs_EoverP_Erg.h5")
    names = h5open(keys, path)
    for k in (
            "Total_Excitation", "Total_Momentum", "Ionization", "Elastic",
            "Momentum_by_ela", "Momentum_by_exc", "Momentum_by_iz",
        )
        @test !(k in names)
    end
    # And the modern set is all still there — a truncated file would also pass the above.
    for k in ("K_iz", "K_diss_iz", "K_exc", "K_mom", "L_ela", "L_exc", "L_tot")
        @test k in names
    end
end

@testsnippet LedgerRAPID begin
    using RAPID2D
    function ledger_RAPID()
        config = SimulationConfig{Float64}(
            NR = 8, NZ = 8, prefilled_gas_pressure = 1.0e-2, R0B0 = 1.0,
            dt = 1.0e-8, snap0D_Δt_s = 1.0, snap2D_Δt_s = 1.0,
        )
        # cleanup = false: the ADIOS2 snapshot writer outlives an auto-cleaned dir
        # (same pitfall documented at random_step_ceiling_test.jl:214) — an atexit-
        # scheduled rm on the default mktempdir() races the writer's own close and
        # aborts the process.
        config.Output_path = mktempdir(; cleanup = false)
        RP = RAPID{Float64}(config)
        initialize!(RP)
        return RP
    end
end

@testitem "RRC loader: every 2026-08 surface is reachable by name" setup = [LedgerRAPID] begin
    using RAPID2D: get_electron_RRC
    RP = ledger_RAPID()
    for s in (
            :K_iz, :K_diss_iz, :K_exc, :K_diss_exc,
            :K_mom, :K_mom_by_ela, :K_mom_by_exc, :K_mom_by_diss_exc,
            :K_mom_by_iz, :K_mom_by_diss_iz,
            :Kerg_ela, :Kerg_exc, :Kerg_diss_exc, :Kerg_tot,
        )
        v = get_electron_RRC(RP, s)
        @test size(v) == size(RP.plasma.Te_eV)
        @test all(isfinite, v)
        @test all(>=(0.0), v)          # every ledger member is non-negative
    end
end

@testitem "RRC loader: Kerg_ela already carries 2me/M, so it is tiny next to Kerg_exc" setup = [LedgerRAPID] begin
    using RAPID2D: get_electron_RRC
    # A guard against the easiest error in this migration: re-applying 2me/M (5.4e-4)
    # to a coefficient that already contains it. If someone did, Kerg_ela would drop
    # three orders of magnitude below any plausible energy sink.
    RP = ledger_RAPID()
    RP.plasma.Te_eV .= 5.0
    RP.fields.E_para_tot .= 30.0
    ela = get_electron_RRC(RP, :Kerg_ela)
    exc = get_electron_RRC(RP, :Kerg_exc)
    @test all(>(0.0), ela)
    # At Te = 5 eV the EXC group dominates but not by more than ~3 decades.
    @test maximum(exc) / maximum(ela) < 1.0e3
end

@testitem "update_RRCs!: the energy ledger is materialized, and it closes" setup = [LedgerRAPID] begin
    using RAPID2D: update_RRCs!, get_electron_RRC
    RP = ledger_RAPID()
    RP.plasma.Te_eV .= 5.0
    RP.fields.E_para_tot .= 30.0
    RP.flags.Atomic_Collision = true
    RP.flags.src = true
    update_RRCs!(RP)

    pla = RP.plasma
    ng = pla.n_H2_gas
    ee = RP.config.constants.ee

    @test all(>(0.0), pla.P_en_exc)
    @test all(>(0.0), pla.P_en_ela)
    @test all(>=(0.0), pla.ν_en_diss_iz)

    # The same closure as the on-disk test, now through the interpolants and the
    # materialization. Interpolation makes it approximate rather than exact.
    # (Not `@. ng * get_electron_RRC(RP, :Kerg_tot)`: `@.` would also dot the RP
    # argument of the function call, and RP has no `Broadcast.broadcastable` of its
    # own, so it falls through to the generic-iterable fallback and fails on
    # `length(RP)`. Call once, then broadcast the multiply.)
    tot = ng .* get_electron_RRC(RP, :Kerg_tot)
    parts = @. pla.P_en_ela + pla.P_en_exc + pla.P_en_diss_exc +
        ee * (15.426 * pla.ν_en_iz + 35.0 * pla.ν_en_diss_iz)
    @test maximum(abs.(parts .- tot) ./ tot) < 1.0e-6
end

@testitem "update_RRCs!: the energy ledger is only materialized under Atomic_Collision" setup = [LedgerRAPID] begin
    using RAPID2D: update_RRCs!
    # Same gating as the momentum frequencies: a run with atomic collisions off must not
    # pay four interpolations per step, and must not leave a stale sink for the energy
    # equation to charge.
    RP = ledger_RAPID()
    RP.flags.Atomic_Collision = false
    RP.flags.src = false
    RP.plasma.P_en_exc .= 1.0e99          # poison
    update_RRCs!(RP)
    @test all(==(1.0e99), RP.plasma.P_en_exc)   # untouched, not silently refreshed
end

@testitem "RRC loader: the 12 eV excitation normalization is gone" setup = [LedgerRAPID] begin
    # The constant is not merely unused, it is unrepresentable: ⟨ΔE⟩_exc runs 0.059 to
    # 9.72 eV across the operating range, a factor 165. Keeping a field for it invites
    # someone to reconstruct P_exc from it again.
    @test !hasfield(RAPID2D.PlasmaConstants{Float64}, :char_exc_erg_eV)
    RP = ledger_RAPID()
    @test !hasfield(typeof(RP.eRRCs), :Total_Excitation)
    @test !hasfield(typeof(RP.plasma), :ν_en_exc_eff)
end

@testitem "RRC ledger: iz_erg_eV matches the cross-section source BD sampled" begin
    # Yoon 2008 section 9: "The best value of the ionization potential of H2 is 15.426 eV".
    # BD's E_IONIZATION_EV carries that value, and the L_tot assembly above is built with
    # it, so a different constant here would break the energy ledger by 0.2 %.
    c = RAPID2D.PlasmaConstants{Float64}()
    @test c.iz_erg_eV == 15.426
end

@testitem "RRC loader: the (T,ud) dissociative-ionization surface is marked legacy" setup = [LedgerRAPID] begin
    # `Dissoc_Ionz` comes from eRRCs_T_ud.h5: a different quantity from K_diss_iz, on
    # different coordinates, from a different data generation, and consumed by nothing.
    # Sharing a plain name with the live channel is a trap, not an alias.
    RP = ledger_RAPID()
    @test !hasfield(typeof(RP.eRRCs), :Dissoc_Ionz)
    @test hasfield(typeof(RP.eRRCs), :Dissoc_Ionz_legacy)
end
