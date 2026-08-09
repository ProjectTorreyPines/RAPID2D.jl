@testitem "T_ud tables: axis 1 is temperature, checked against physics" begin
    using RAPID2D: load_electron_RRCs, H2_Ion_RRCs, RRC_T_ud

    # A shape assertion cannot catch this — both axes have the same length, which
    # is why the transposition survived undetected. So check a value the physics
    # pins: electron-impact ionization has a 15.4 eV threshold, so the Maxwellian
    # average must underflow to zero at 0.001 eV and rise through the tens of eV.
    # Read the other way round the same surface reports 3.6e-14 m³/s at 19 meV,
    # which would need exp(−15.4/0.019).
    eRRCs = load_electron_RRCs()
    iz = eRRCs.Dissoc_Ionz            # the (T, u_d) ionization surface in that file
    T, u = iz.T_eV, iz.ud_para
    @test size(iz.raw_data) == (length(T), length(u))

    # Resolve the grid indices first: a `findfirst` that misses returns `nothing` and
    # would error on indexing rather than say which assumption broke.
    i_warm, i_below = findfirst(≥(50.0), T), findfirst(≥(0.1), T)
    @test i_warm !== nothing && i_below !== nothing   # the grid spans 1 meV … 3 keV

    @test iz.raw_data[1, 1] == 0.0                    # T = 1 meV, zero drift
    @test iz.raw_data[i_warm, 1] > 1.0e-15
    # Monotone rise across the threshold, which the transposed read does not give.
    @test iz.raw_data[i_below, 1] < 1.0e-20

    # The ion file's orientation is NOT re-checked here on purpose. Every physical
    # signature available for it — Particle_Exchange's 1.64–20.1 eV support,
    # Charge_Exchange vanishing below 4.9 eV — is a fit-domain guard scheduled for
    # removal in the ion cross-section overhaul, so pinning one would make this test
    # fail on the day the data is corrected and read as a regression. The contract
    # (axis_order, dimensions) is asserted below and survives any data; the physics
    # lives in `scripts/check_ion_rrc_tables.jl`, which recomputes ⟨σv⟩ from the cross
    # sections and is meant to be re-run against each new file.
end

@testitem "T_ud tables: the ion file's orientation is pinned by its own values" begin
    using RAPID2D: H2_Ion_RRCs

    # The attribute is a declaration, and a regenerated file could carry it while
    # still being transposed. These three entries are OFF-DIAGONAL, so transposing
    # swaps each with a different number — 7.19e-16 becomes 7.82e-16 and so on.
    # That is the whole point: the contract tests above check what the file SAYS,
    # this one checks what it HOLDS.
    #
    # Pinned to the shipped data, deliberately. A legitimate regeneration must fail
    # here and be re-measured — see RRC_data/README.md, and run
    # `scripts/check_ion_rrc_tables.jl` before trusting the new numbers.
    iRRCs = H2_Ion_RRCs(
        joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "iRRCs_T_ud.h5")
    )
    for (channel, i, j, expected) in (
            (:Elastic, 30, 70, 7.1930558493e-16),          # T = 0.0802 eV, u = 6.82e4 m/s
            (:Elastic, 60, 10, 1.2454786702e-15),          # T = 7.477  eV, u = 8.89e3 m/s
            (:Charge_Exchange, 80, 20, 1.0738585508e-14),  # T = 153.8  eV, u = 1.88e4 m/s
        )
        A = getfield(iRRCs, channel).raw_data
        @test A[i, j] ≈ expected rtol = 1.0e-9
        @test A[i, j] != A[j, i]        # off-diagonal, so a transpose is visible here
    end
end

@testitem "T_ud tables: the reader follows the file's declared axis order" begin
    # h5open/attrs via RAPID2D: HDF5 is its dependency, not the test environment's.
    using RAPID2D: H2_Ion_RRCs, h5open, attrs

    # Three ways of storing the same surface must load identically: unstamped
    # (legacy, transposed), explicitly stamped legacy, and stamped in the
    # constructor's order. That is what lets the cross-section overhaul ship a
    # correctly-oriented file without touching a line of this.
    src = joinpath(dirname(dirname(pathof(RAPID2D))), "RRC_data", "iRRCs_T_ud.h5")
    T, u, surf = h5open(src) do f
        (
            read(f, "T_eV"), read(f, "ud_para"),
            Dict(k => read(f, k) for k in keys(f) if k ∉ ("T_eV", "ud_para")),
        )
    end

    function write_variant(; order, transpose)
        path = joinpath(mktempdir(), "iRRCs_T_ud.h5")
        h5open(path, "w") do f
            f["T_eV"] = T
            f["ud_para"] = u
            for (k, v) in surf
                f[k] = transpose ? permutedims(v) : v
            end
            order === nothing || (attrs(f)["axis_order"] = order)
        end
        return H2_Ion_RRCs(path)
    end

    legacy_unstamped = write_variant(; order = nothing, transpose = false)
    legacy_stamped = write_variant(; order = "ud_para,T_eV", transpose = false)
    corrected = write_variant(; order = "T_eV,ud_para", transpose = true)

    for ch in (:Elastic, :Charge_Exchange, :Particle_Exchange)
        a = getfield(legacy_unstamped, ch).raw_data
        @test a == getfield(legacy_stamped, ch).raw_data
        @test a == getfield(corrected, ch).raw_data
    end

    # An axis_order nobody has defined is a mistake, not a third layout to guess.
    bad = joinpath(mktempdir(), "iRRCs_T_ud.h5")
    h5open(bad, "w") do f
        f["T_eV"] = T
        f["ud_para"] = u
        for (k, v) in surf
            f[k] = v
        end
        attrs(f)["axis_order"] = "energy,angle"
    end
    @test_throws ArgumentError H2_Ion_RRCs(bad)
end

@testitem "RRC_T_ud: a genuinely transposed surface raises on construction" begin
    using RAPID2D: RRC_T_ud

    # Non-square, so the dimension check has something to see. The shipped files
    # are square and this is what they could not do for themselves.
    T = collect(range(0.1, 10.0, 7))
    u = collect(range(0.0, 1.0e5, 11))
    ok = [t * v for t in T, v in u]
    @test RRC_T_ud(T, u, ok) isa RRC_T_ud
    @test_throws DimensionMismatch RRC_T_ud(T, u, permutedims(ok))
end
