@testitem "RRC_EoverP_Erg: ∂K/∂Ē is exact on a table linear in Ē" begin
    using RAPID2D: RRC_EoverP_Erg

    # Bilinear interpolation reproduces a function linear in Ē EXACTLY, on any
    # grid. So a synthetic table K = a + b·Ē pins ∂K/∂Ē = b to machine precision
    # and gives the derivative path a reference that finite differencing cannot
    # match (FD tops out near 1e-6 on truncation alone).
    EoverP = [1.0, 3.0, 10.0, 30.0, 100.0]
    Erg_eV = [0.01, 0.1, 1.0, 10.0, 100.0]
    a, b = 2.0e-15, 3.5e-16
    data = [a + b * E for _ in EoverP, E in Erg_eV]
    rrc = RRC_EoverP_Erg(EoverP, Erg_eV, data)

    q_eop = [1.5, 7.0, 55.0, 99.0]
    q_erg = [0.05, 0.5, 5.0, 50.0]

    @test rrc.itp((q_eop, q_erg)) ≈ a .+ b .* q_erg rtol = 1.0e-14
    @test rrc.dK_dĒ((q_eop, q_erg)) ≈ fill(b, 4) rtol = 1.0e-12
end

@testitem "RRC_EoverP_Erg: the derivative clamps on E/p and refuses to guess on Ē" begin
    using RAPID2D: RRC_EoverP_Erg

    # Two different jobs on the two axes, and the asymmetry is not stylistic.
    #
    # E/p does not depend on Tₑ at all, so ∂/∂Tₑ never sees it — clamping there is
    # both correct and REQUIRED, because the value path clamps too (below the
    # table's minimum E/p the rate relaxes to the room-T Maxwellian bottom row;
    # E/p = 0 means no field, not no collisions). A NoExtrap on that axis would
    # throw on ordinary low-field cells.
    #
    # Ē is the axis ∂/∂Tₑ rides. Outside the table the value is frozen, so the
    # true derivative is 0 — but FastInterpolations' ClampExtrap clamps the
    # COORDINATE and then returns the boundary cell's one-sided slope, which is
    # not 0. That is an upstream bug (the value path is unaffected). Until it is
    # fixed, refusing to answer beats answering wrongly: a Jacobian claiming Tₑ
    # dependence where the value has none is silent, and a DomainError is not.
    EoverP = [1.0, 10.0, 100.0]
    Erg_eV = [0.1, 1.0, 10.0]
    data = [E for _ in EoverP, E in Erg_eV]
    rrc = RRC_EoverP_Erg(EoverP, Erg_eV, data)

    # In range: both paths agree with the analytic slope of this cell.
    @test rrc.dK_dĒ((5.0, 5.0)) ≈ 1.0 rtol = 1.0e-12

    # E/p out of range on BOTH sides: clamped, no throw, same as the value path.
    @test rrc.itp((0.5, 5.0)) == rrc.itp((1.0, 5.0))
    @test rrc.dK_dĒ((0.5, 5.0)) ≈ rrc.dK_dĒ((1.0, 5.0))
    @test rrc.dK_dĒ((1.0e4, 5.0)) ≈ rrc.dK_dĒ((100.0, 5.0))

    # Ē out of range: the value still clamps (physics unchanged), the derivative
    # throws.
    @test rrc.itp((5.0, 1.0e-3)) == rrc.itp((5.0, 0.1))
    @test rrc.itp((5.0, 1.0e3)) == rrc.itp((5.0, 10.0))
    @test_throws DomainError rrc.dK_dĒ((5.0, 1.0e-3))
    @test_throws DomainError rrc.dK_dĒ((5.0, 1.0e3))
end

@testitem "RRC_EoverP_Erg: the derivative is a batch, in-place, allocation-free path" begin
    using RAPID2D: RRC_EoverP_Erg

    # update_RRCs! evaluates a whole grid per step. The derivative must take the
    # same SoA tuple the value path already builds, and write into a preallocated
    # buffer — otherwise the "one extra table evaluation" of the design note
    # becomes an allocation per node per step.
    EoverP = collect(range(1.0, 100.0, 40))
    Erg_eV = collect(10 .^ range(-2, 2, 60))
    data = [sqrt(p) * E^2 for p in EoverP, E in Erg_eV]
    rrc = RRC_EoverP_Erg(EoverP, Erg_eV, data)

    n = 400
    qp = fill(50.0, n)
    qe = fill(1.0, n)
    out = zeros(n)

    rrc.dK_dĒ(out, (qp, qe))
    @test all(out .≈ rrc.dK_dĒ((qp, qe)))
    @test all(>(0), out)

    # `dK_dĒ` is an untyped field, exactly like `itp`, so each CALL pays a fixed
    # dynamic-dispatch box. What must not happen is allocation per NODE — that is
    # what would turn the design note's "one extra table evaluation" into a real
    # cost. Measure the scaling, not the constant, and hold the derivative path to
    # the value path's standard rather than to an absolute that `itp` also misses.
    alloc(f, out, q) = (f(out, q); @allocated f(out, q))
    small = alloc(rrc.dK_dĒ, out, (qp, qe))
    big_q = (fill(50.0, 10n), fill(1.0, 10n))
    big = alloc(rrc.dK_dĒ, zeros(10n), big_q)
    @test big == small                                   # O(1) in the node count
    @test small == alloc(rrc.itp, out, (qp, qe))         # no worse than the value path
end

@testitem "RRC_EoverP_Erg: real tables carry a usable derivative on both axes' interior" begin
    using RAPID2D: load_electron_RRCs

    # The four surfaces update_RRCs! reads are exactly the four ∂P/∂Tₑ needs, so
    # nothing new is looked up — only differentiated. Check the wiring survives
    # the real loader and the real (201 × 300) grid.
    eRRCs = load_electron_RRCs()
    for name in (:K_iz, :K_mom, :K_mom_by_ela, :Total_Excitation)
        rrc = getfield(eRRCs, name)
        Ē_lo, Ē_hi = extrema(rrc.Erg_eV)
        p_lo, p_hi = extrema(rrc.EoverP)

        # Sampled well inside both axes, and against a central difference of the
        # interpolant itself — same object, so this checks the derivative wiring,
        # not the table's physics.
        for p in range(p_lo * 1.5, p_hi * 0.7, 5), Ē in 10 .^ range(log10(Ē_lo * 10), log10(Ē_hi * 0.1), 7)
            h = 1.0e-4 * Ē
            fd = (rrc.itp((p, Ē + h)) - rrc.itp((p, Ē - h))) / (2h)
            an = rrc.dK_dĒ((p, Ē))
            @test isapprox(an, fd; rtol = 1.0e-6, atol = 1.0e-24)
        end

        # And E/p below the table still clamps rather than throwing, because that
        # is an ordinary low-field cell, not an error.
        @test rrc.dK_dĒ((p_lo / 10, 1.0)) ≈ rrc.dK_dĒ((p_lo, 1.0))
    end
end
