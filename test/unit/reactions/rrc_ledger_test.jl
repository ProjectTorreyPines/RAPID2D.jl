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

@testitem "RRC ledger: the deprecated Total_Excitation alias is corrupt in the cold band" begin
    using RAPID2D.HDF5
    # BD's cold-band fill wrote the raw EXC-group rate K_exc into the legacy alias below
    # 0.0388 eV, where the electronic channels it is supposed to encode (thresholds >=
    # 8.9 eV) are identically zero. Pinned so the defect is visible at the data level and
    # so deleting Total_Excitation in Task B5 is provably a removal of something broken.
    path = joinpath(pkgdir(RAPID2D), "RRC_data", "eRRCs_EoverP_Erg.h5")
    E, te, ke = h5open(path) do f
        read(f, "Erg_eV"), read(f, "Total_Excitation"), read(f, "K_exc")
    end
    cold = findall(<(0.0388), E)
    warm = findall(>(0.05), E)
    @test any(!iszero, te[:, cold])          # nonzero where electronic excitation cannot be
    @test te[:, cold] == ke[:, cold]         # and it is bit-identical to the EXC group rate
    @test te[:, warm] != ke[:, warm]         # above the cold band the alias is its own thing
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
