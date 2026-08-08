# Field-line following: what a trace publishes, and why it ended.
#
# The contract these tests pin is that `Lpol` and `Lc` answer DIFFERENT questions:
#
#   Lpol  distance TRAVELLED in the poloidal projection
#   Lc    the geometric bound along B — how far a step can go before the geometry
#         stops being new: a wall, one circuit on a closed line, or 2πR at a Bpol null
#
# `Lc` is `Inf` from exactly one source, `:trace_limit`, and it means the measurement
# FAILED rather than that the line is unbounded. Every other outcome publishes a length
# it measured, and `validate_field_line_terminations!` raises on the one that does not.
# That is the whole contract, and what it guards is real: a trace that ran out of budget
# used to publish the partial distance it had covered, which downstream is
# indistinguishable from a measured one, so a ceiling built on it would silently use a
# made-up length.

@testsnippet FLFFields begin
    using RAPID2D: my_interpolation, trace_single_field_line,
        flf_analysis_field_lines_rz_plane!, FieldLineFollowingResult, FLF_TERMINATIONS

    const R1D = collect(range(0.5, 2.5; length = 41))
    const Z1D = collect(range(-1.0, 1.0; length = 41))

    "Uniform vertical poloidal field: every line runs straight up/down. Bpol/Bϕ is fixed,
    so the ratio Lc/Lpol = Btot/Bpol is known in closed form."
    function straight_field(; BZ = 0.1, Bϕ = 1.0)
        n = (length(R1D), length(Z1D))
        return (
            my_interpolation(R1D, Z1D, zeros(n...)),
            my_interpolation(R1D, Z1D, fill(BZ, n...)),
            my_interpolation(R1D, Z1D, fill(Bϕ, n...)),
        )
    end

    "Concentric poloidal circles about (Rc, Zc): B_pol ∝ (-(Z-Zc), (R-Rc)) closes in 2π."
    function circular_arrays(; Rc = 1.5, Zc = 0.0, ω = 1.0, Bϕ = 1.0)
        BR = [-ω * (z - Zc) for _ in R1D, z in Z1D]
        BZ = [ω * (r - Rc) for r in R1D, _ in Z1D]
        return BR, BZ, fill(Bϕ, length(R1D), length(Z1D))
    end

    "A wall checker for an axis-aligned box, independent of the grid indexing under test.
    Returns `true` where the point is still valid plasma, matching `wall_checker`'s sense."
    box_wall(Rlo, Rhi, Zlo, Zhi) = (R, Z) -> (Rlo <= R <= Rhi) && (Zlo <= Z <= Zhi)
end

@testitem "Wall checker: leaving on one axis is not re-entry on the other" begin
    using RAPID2D: is_in_wall_by_cell_state

    NR, NZ = 4, 3
    cell_state = trues(NR, NZ)          # all plasma; only geometry can make this false
    Rmin, Zmin = 0.0, 0.0
    inv_dR = inv_dZ = 1.0               # so index == floor(coordinate) + 1

    # Interior sanity first — without these the negatives below would prove nothing.
    @test is_in_wall_by_cell_state(0.5, 0.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)
    @test is_in_wall_by_cell_state(3.5, 2.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)

    # R past the RIGHT edge: Rid = 5, Zid = 2 ⟹ nid = 1*4 + 5 = 9, a perfectly valid
    # index naming (1, 3). Checking only the linear index returned `true` here, so a
    # trace could leave the vessel sideways and be told it was still inside.
    @test !is_in_wall_by_cell_state(4.5, 1.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)
    # R past the LEFT edge: Rid = 0, Zid = 2 ⟹ nid = 4, naming (4, 1).
    @test !is_in_wall_by_cell_state(-0.5, 1.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)

    # Z beyond either end always fell outside the linear range; pin it so the fix cannot
    # regress the half that already worked.
    @test !is_in_wall_by_cell_state(1.5, 3.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)
    @test !is_in_wall_by_cell_state(1.5, -0.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)

    # A real wall cell inside the domain still reads as wall.
    cell_state[2, 2] = false
    @test !is_in_wall_by_cell_state(1.5, 1.5, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state)
end

@testitem "FLF trace: a wall hit publishes the distance to the wall" setup = [FLFFields] begin
    iBR, iBZ, iBϕ = straight_field(; BZ = 0.1, Bϕ = 1.0)
    wall = box_wall(0.6, 2.4, -0.8, 0.8)

    # From Z = 0 upward the wall is 0.8 m away in the poloidal plane. Lc is measured
    # ALONG B, so it is longer by Btot/Bpol = √(1 + (Bϕ/Bpol)²) = √101 ≈ 10.05. This is
    # the only test that pins the two lengths apart on the same trace.
    fwd = trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 100_000, 10.0, wall)
    @test fwd.termination === :wall
    @test fwd.hit_wall
    @test isapprox(fwd.Lpol, 0.8; rtol = 5.0e-3)
    @test isapprox(fwd.Lc, 0.8 * sqrt(101); rtol = 5.0e-3)

    bwd = trace_single_field_line(1.5, 0.0, -1, iBR, iBZ, iBϕ, 0.001, 100_000, 10.0, wall)
    @test bwd.termination === :wall
    @test isapprox(bwd.Lpol, 0.8; rtol = 5.0e-3)

    # An asymmetric start proves the two directions are measured independently rather
    # than one being copied onto the other.
    off = trace_single_field_line(1.5, 0.4, 1, iBR, iBZ, iBϕ, 0.001, 100_000, 10.0, wall)
    @test isapprox(off.Lpol, 0.4; rtol = 1.0e-2)
end

@testitem "FLF trace: an exhausted budget publishes Inf, not how far it got" setup = [FLFFields] begin
    iBR, iBZ, iBϕ = straight_field(; BZ = 0.1, Bϕ = 1.0)
    never = box_wall(-Inf, Inf, -Inf, Inf)

    # `Lpol > max_Lpol` and `max_steps` are ONE physical condition — the budget ran out —
    # and both must refuse to publish the partial distance. That partial value is exactly
    # what used to be indistinguishable from a wall distance downstream.
    by_length = trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 100_000, 0.05, never)
    @test by_length.termination === :trace_limit
    @test by_length.Lc == Inf
    @test !by_length.hit_wall
    # Lpol keeps the travelled distance: a caller can still see how far it got.
    @test isapprox(by_length.Lpol, 0.05; atol = 2.0e-3)

    by_steps = trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 10, 10.0, never)
    @test by_steps.termination === :trace_limit
    @test by_steps.Lc == Inf
    @test by_steps.steps == 10
end

@testitem "FLF trace: a Bpol null is a toroidal circle, not an unbounded line" setup = [FLFFields] begin
    using RAPID2D: my_interpolation, trace_single_field_line

    n = (length(R1D), length(Z1D))
    zero_pol = my_interpolation(R1D, Z1D, zeros(n...))
    iBϕ = my_interpolation(R1D, Z1D, fill(1.0, n...))
    never = box_wall(-Inf, Inf, -Inf, Inf)

    # `Bpol = 0` stops the tracer because the poloidal projection cannot move — but the
    # FIELD LINE is perfectly well defined there: purely toroidal, so a closed circle of
    # circumference 2πR. Publishing `Inf` would say "unbounded", which is not what a
    # vanishing poloidal field means, and would hand the transport ceiling an excuse to
    # switch off at a point where the geometry is known exactly.
    for R0 in (1.0, 1.5, 2.3)
        res = trace_single_field_line(R0, 0.0, 1, zero_pol, zero_pol, iBϕ, 0.001, 100, 10.0, never)
        @test res.termination === :null
        @test res.Lc ≈ 2π * R0
        @test res.steps == 0
    end
end

@testitem "FLF trace: every exit sets a termination from the published set" setup = [FLFFields] begin
    iBR, iBZ, iBϕ = straight_field()
    wall = box_wall(0.6, 2.4, -0.8, 0.8)
    never = box_wall(-Inf, Inf, -Inf, Inf)

    # The defensive `:unset` default must never survive a real trace. Enumerate the
    # reachable exits and require each to name itself; a construction site that forgets
    # its keyword shows up here rather than as a mystery `Inf` in transport.
    results = (
        trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 100_000, 10.0, wall),
        trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 100_000, 0.05, never),
        trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 10, 10.0, never),
    )
    for res in results
        @test res.termination in FLF_TERMINATIONS
        @test res.termination !== :unset
        # THE CONTRACT IN ONE LINE, over every exit at once. `Inf` has exactly one
        # source and one meaning — the measurement failed — and every other outcome
        # publishes the length it measured. An unknown distance must never look like a
        # measured one, which is the whole reason this contract exists.
        @test isfinite(res.Lc) == (res.termination !== :trace_limit)
    end
end

@testitem "FLF assembly: a closed line publishes its circuit length" setup = [FLFFields] begin
    BR, BZ, Bϕ = circular_arrays(; Rc = 1.5, Zc = 0.0, ω = 1.0)
    NR, NZ = length(R1D), length(Z1D)
    flf = FieldLineFollowingResult{Float64}(NR, NZ)

    # A wall inside the interpolation domain, as on a real grid. Traces reaching the
    # domain edge are handled — `my_interpolation` clamps — but this fixture is about
    # closure, and a wall keeps the circles it cares about away from that machinery.
    cell_state = trues(NR, NZ)
    cell_state[1:3, :] .= false
    cell_state[(end - 2):end, :] .= false
    cell_state[:, 1:3] .= false
    cell_state[:, (end - 2):end] .= false

    flf_analysis_field_lines_rz_plane!(flf, R1D, Z1D, BR, BZ, Bϕ, cell_state)

    # A node on the circle of radius 0.3 about the centre, comfortably inside the domain.
    i = argmin(abs.(R1D .- 1.8))
    j = argmin(abs.(Z1D .- 0.0))
    @test flf.is_closed[i, j]

    # THE ASSEMBLY IS WHAT IS UNDER TEST HERE. When the forward trace closes, the
    # backward result is synthesised rather than traced — the one construction site that
    # never runs the tracer. If it forgets its termination or inherits the forward `Lc`,
    # only this catches it.
    @test flf.termination_forward[i, j] === :closed
    @test flf.termination_backward[i, j] === :closed

    # A closed line publishes its CIRCUIT length, not `Inf`. The geometry repeats after
    # one circuit, so a step longer than that reaches nowhere new — a measured bound,
    # which is the whole test for whether `Lc` may be finite. On this field
    # `Bpol = ω·r = 0.3` and `Bϕ = 1`, so the circuit along B is longer than the
    # poloidal one by Btot/Bpol = √(0.09 + 1)/0.3.
    circuit_pol = 2π * 0.3
    @test isapprox(flf.Lpol_tot[i, j], circuit_pol; rtol = 5.0e-2)
    @test isapprox(flf.Lc_forward[i, j], circuit_pol * sqrt(1.09) / 0.3; rtol = 5.0e-2)
    # Both directions carry the full circuit — that is what makes `wall_step_ceiling`
    # return ¼v̄(L + L) = ½v̄L, the same form a symmetric pair of walls produces.
    @test flf.Lc_backward[i, j] == flf.Lc_forward[i, j]
    # ...while the TOTAL names the length of the line, and a line is one circuit long.
    @test flf.Lc_tot[i, j] == flf.Lc_forward[i, j]
    @test flf.Lpol_tot[i, j] == flf.Lpol_forward[i, j]
end

@testitem "FLF assembly: Lc_tot is a plain sum of two wall distances" setup = [FLFFields] begin
    n = (length(R1D), length(Z1D))
    BR, BZ, Bϕ = zeros(n...), fill(0.1, n...), fill(1.0, n...)
    NR, NZ = n
    flf = FieldLineFollowingResult{Float64}(NR, NZ)

    # Carve a wall band so vertical lines terminate inside the domain rather than at its
    # edge; both directions then genuinely reach a wall.
    cell_state = trues(NR, NZ)
    cell_state[:, Z1D .< -0.5] .= false
    cell_state[:, Z1D .> 0.5] .= false

    flf_analysis_field_lines_rz_plane!(flf, R1D, Z1D, BR, BZ, Bϕ, cell_state)

    i = argmin(abs.(R1D .- 1.5))
    j = argmin(abs.(Z1D .- 0.0))
    @test flf.termination_forward[i, j] === :wall
    @test flf.termination_backward[i, j] === :wall
    @test isfinite(flf.Lc_forward[i, j])
    @test isfinite(flf.Lc_backward[i, j])
    # No closed-line special case survives: the total is the two distances added. The old
    # code overwrote this whenever `is_closed` was set, which under the wall-distance
    # contract would have thrown away one direction's answer.
    @test flf.Lc_tot[i, j] === flf.Lc_forward[i, j] + flf.Lc_backward[i, j]

    # Each direction on its own satisfies Lc/Lpol = Btot/Bpol = √(1 + (Bϕ/Bpol)²). This
    # is the real invariant; forward and backward are NOT required to match, because
    # `is_in_wall_by_cell_state` resolves a `floor`-indexed CELL and the band above lands
    # half a cell differently on the two sides. That asymmetry is the fixture's, not the
    # code's, and asserting symmetry would have been a test of the fixture.
    @test isapprox(flf.Lc_forward[i, j] / flf.Lpol_forward[i, j], sqrt(101); rtol = 1.0e-3)
    @test isapprox(flf.Lc_backward[i, j] / flf.Lpol_backward[i, j], sqrt(101); rtol = 1.0e-3)
    @test isapprox(flf.Lc_tot[i, j] / flf.Lpol_tot[i, j], sqrt(101); rtol = 1.0e-3)
end

@testsnippet FLFStartup begin
    using RAPID2D: SimulationConfig, SimulationFlags, RAPID, initialize!,
        validate_field_line_terminations!, flf_analysis_field_lines_rz_plane!

    "A startup-like device: strong toroidal field, weak poloidal, open field lines."
    function startup_RP(; NR = 30, NZ = 40)
        config = SimulationConfig{Float64}(
            NR = NR, NZ = NZ, R_min = 0.8, R_max = 2.2, Z_min = -1.2, Z_max = 1.2,
            dt = 1.0e-6, t_end_s = 1.0e-5, R0B0 = 1.0,
            Dpara0 = 0.0, Dperp0 = 0.0, prefilled_gas_pressure = 5.0e-3,
            wall_R = [1.0, 2.0, 2.0, 1.0], wall_Z = [-1.0, -1.0, 1.0, 1.0],
        )
        config.Output_path = mktempdir(; cleanup = false)   # the writer outlives an auto-cleaned dir
        RP = RAPID{Float64}(config)
        RP.flags = SimulationFlags{Float64}(Ampere = false, Gas_evolve = false)
        initialize!(RP)
        return RP
    end
end

@testitem "FLF lifecycle: lengths exist before transport is first built" setup = [FLFStartup] begin
    RP = startup_RP()

    # ASSERTED IMMEDIATELY AFTER `initialize!`, with no manual refresh. If the FLF call
    # were placed after `initialize_plasma_and_transport!` — or omitted, as it was — the
    # first transport build would read `Lc = 0` and cap D∥ at zero across the whole grid.
    inw = RP.G.nodes.in_wall_nids
    tf, tb = RP.flf.termination_forward, RP.flf.termination_backward

    @test !isempty(inw)
    @test all(n -> tf[n] === :wall, inw)
    @test all(n -> tb[n] === :wall, inw)
    @test all(n -> isfinite(RP.flf.Lc_forward[n]) && RP.flf.Lc_forward[n] > 0, inw)
    @test all(n -> isfinite(RP.flf.Lc_backward[n]) && RP.flf.Lc_backward[n] > 0, inw)

    # Lc is measured along B and Lpol in the poloidal projection, so on a strongly
    # toroidal device the ratio is large. This distinguishes a real parallel distance
    # from the poloidal one the ion channel used to substitute for it.
    i = first(inw)
    @test RP.flf.Lc_tot[i] > 10 * RP.flf.Lpol_tot[i]

    # The turbulent channel's own length must not have been dragged forward by this:
    # `estimate_electrostatic_field_effects!` is what populates it, and `initialize!`
    # does not call that.
    @test all(iszero, RP.transport.L_mixing)
end

@testitem "FLF validation: an exhausted budget is rejected, not silently believed" setup = [FLFStartup] begin
    RP = startup_RP()

    # `:trace_limit` means the distance is UNKNOWN. Letting it read as `Lc = Inf` would
    # switch the geometric ceiling off on the strength of a measurement that failed —
    # the precise failure the ceiling exists to prevent.
    RP.flf.termination_forward[first(RP.G.nodes.in_wall_nids)] = :trace_limit
    err = try
        validate_field_line_terminations!(RP)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    # The message must carry its own escape hatch: throwing kills `initialize!`, and
    # `max_Lpol` is a derived default a caller has no obvious way to raise.
    @test occursin("max_Lpol", err.msg)
    @test occursin(string(RP.flf.max_Lpol), err.msg)

    # An untraced node is a different failure with the same consequence, and is caught.
    RP2 = startup_RP()
    RP2.flf.termination_backward[first(RP2.G.nodes.in_wall_nids)] = :unset
    @test_throws ArgumentError validate_field_line_terminations!(RP2)
end

@testitem "FLF validation: a null or half-closed pair warns, it does not throw" setup = [FLFStartup] begin
    RP = startup_RP()
    nid = first(RP.G.nodes.in_wall_nids)

    # Both leave the ceiling off at the affected node, which is what this code did
    # everywhere before the ceiling existed — so neither is a regression worth killing
    # a run over. The half-closed case in particular is expected to be reachable: the
    # detector accumulates turning angle rather than testing return-to-start.
    RP.flf.termination_forward[nid] = :null
    @test (@test_logs (:warn,) match_mode = :any validate_field_line_terminations!(RP)) === RP.flf

    RP2 = startup_RP()
    RP2.flf.termination_forward[first(RP2.G.nodes.in_wall_nids)] = :closed
    @test (@test_logs (:warn,) match_mode = :any validate_field_line_terminations!(RP2)) === RP2.flf
end

@testitem "FLF totals: a half-closed pair keeps both measurements" begin
    using RAPID2D: FieldLineFollowingResult

    # Three nodes, three regimes. The collapse of totals onto one direction is only
    # valid when the two directions retraced the SAME circuit — i.e. both are :closed.
    # Gating it on "either is closed" stomped the backward direction's real measurement
    # at exactly the half-closed nodes the validator's own warning declares reachable
    # (the closure detector accumulates turning angle, so a strongly curved open line
    # can trip it in one direction only). Lpol_tot is not a diagnostic: it feeds
    # L_mixing and through it the turbulent channel.
    flf = FieldLineFollowingResult{Float64}(3, 1)

    # node 1 — genuinely closed (backward synthesized from forward, as assembly does)
    flf.termination_forward[1, 1] = :closed
    flf.termination_backward[1, 1] = :closed
    flf.Lpol_forward[1, 1] = 2.0
    flf.Lpol_backward[1, 1] = 2.0
    flf.Lc_forward[1, 1] = 20.0
    flf.Lc_backward[1, 1] = 20.0
    flf.is_closed[1, 1] = true

    # node 2 — half-closed: forward reached a wall, backward tripped the detector
    flf.termination_forward[2, 1] = :wall
    flf.termination_backward[2, 1] = :closed
    flf.Lpol_forward[2, 1] = 0.5
    flf.Lpol_backward[2, 1] = 2.0
    flf.Lc_forward[2, 1] = 5.0
    flf.Lc_backward[2, 1] = 20.0
    flf.is_closed[2, 1] = true          # the assembly loop flags either-direction

    # node 3 — ordinary open line
    flf.termination_forward[3, 1] = :wall
    flf.termination_backward[3, 1] = :wall
    flf.Lpol_forward[3, 1] = 0.4
    flf.Lpol_backward[3, 1] = 0.6
    flf.Lc_forward[3, 1] = 4.0
    flf.Lc_backward[3, 1] = 6.0

    RAPID2D._finalize_total_lengths!(flf)

    # Both-closed: one circuit, not two.
    @test flf.Lpol_tot[1, 1] == 2.0
    @test flf.Lc_tot[1, 1] == 20.0

    # Half-closed: NOTHING is overwritten. Backward's circuit survives, and the totals
    # are plain sums of what was actually measured.
    @test flf.Lpol_backward[2, 1] == 2.0
    @test flf.Lpol_tot[2, 1] == 2.5
    @test flf.Lc_tot[2, 1] == 25.0

    # Open: plain sums.
    @test flf.Lpol_tot[3, 1] == 1.0
    @test flf.Lc_tot[3, 1] == 10.0
end

@testitem "FLF trace: a mid-trace null publishes walked distance + 2πR" setup = [FLFFields] begin
    using RAPID2D: my_interpolation, trace_single_field_line

    # LINEAR interpolants, deliberately: the zero region must be EXACTLY zero, and a
    # cubic through step data overshoots into small nonzero values there.
    n = (length(R1D), length(Z1D))
    # Nodes at Z ≤ 0.25 carry BZ = 0.1, nodes at Z ≥ 0.30 carry 0. Linear interpolation
    # ramps across the single cell in between, so a trace climbing in +Z sees Bpol > 0
    # up to Z = 0.30 and exactly 0 from there on.
    BZ = [z < 0.275 ? 0.1 : 0.0 for _ in R1D, z in Z1D]
    iBR = my_interpolation(R1D, Z1D, zeros(n...); method = :linear)
    iBZ = my_interpolation(R1D, Z1D, BZ; method = :linear)
    iBϕ = my_interpolation(R1D, Z1D, zeros(n...); method = :linear)   # Btot = Bpol
    never = box_wall(-Inf, Inf, -Inf, Inf)

    res = trace_single_field_line(1.5, 0.0, 1, iBR, iBZ, iBϕ, 0.001, 10_000, 5.0, never)

    # The trace walks ≈ 0.30 m straight up and then runs out of poloidal field. That is
    # a null termination at the last position reached — NOT a wall (nothing was hit)
    # and NOT a crash. The RK4 stages divide by Bpol, so the step that pokes its
    # lookahead into the zero region comes back NaN; before the guard this NaN walked
    # into the wall checker and was misread as a wall hit (any comparison with NaN is
    # false), and the production cell-state checker would have thrown InexactError.
    @test res.termination === :null
    @test !res.hit_wall
    @test res.steps > 250
    @test isfinite(res.Lc)
    # Bϕ = 0 makes Lc accumulate exactly the poloidal distance, so the oracle is
    # closed-form: what was walked, plus the toroidal circle at the point of arrival.
    @test isapprox(res.Lc, 0.3 + 2π * 1.5; rtol = 1.0e-2)
    @test isapprox(res.final_R, 1.5; atol = 1.0e-6)
    @test 0.29 < res.final_Z < 0.302
end

@testitem "FLF validation: a step-loop refresh degrades gracefully instead of dying" setup = [FLFStartup] begin
    using RAPID2D: validate_field_line_terminations!

    # THE OPERATIONAL CONTRACT. At initialize! a failed trace must throw — a run that
    # starts from an unmeasured geometry is wrong from step 0. But the same validator
    # also runs on every periodic FLF refresh (workflows.jl, FLF_nstep), where the
    # evolving field can wiggle one trace past its budget at step N of a long run.
    # Killing that run helps nobody: non-strict mode warns and substitutes a
    # CONSERVATIVE length — the partial poloidal distance the tracer actually walked,
    # scaled by the node's Btot/Bpol. That understates the true distance to the wall
    # (the trace was still going when the budget ran out), so the ceiling it produces
    # is tighter than truth, never looser.
    RP = startup_RP()
    nid = first(RP.G.nodes.in_wall_nids)
    RP.flf.termination_forward[nid] = :trace_limit
    RP.flf.Lc_forward[nid] = Inf
    RP.flf.Lpol_forward[nid] = 0.7                     # the partial the tracer kept
    Bpol = hypot(RP.fields.BR[nid], RP.fields.BZ[nid])
    expected = 0.7 * RP.fields.Btot[nid] / Bpol

    @test_logs (:warn,) match_mode = :any validate_field_line_terminations!(RP; strict = false)
    @test RP.flf.Lc_forward[nid] ≈ expected
    @test isfinite(RP.flf.Lc_forward[nid])
    # The other direction's real measurement is untouched, and the totals re-close.
    @test RP.flf.termination_backward[nid] === :wall
    @test RP.flf.Lc_tot[nid] ≈ RP.flf.Lc_forward[nid] + RP.flf.Lc_backward[nid]
    # The record of WHAT HAPPENED is kept: the fallback is a substituted length, not a
    # reclassification of the trace.
    @test RP.flf.termination_forward[nid] === :trace_limit

    # `:unset` is a code bug — the loop failed to visit a node — and no operational
    # mode makes that tolerable.
    RP2 = startup_RP()
    RP2.flf.termination_backward[first(RP2.G.nodes.in_wall_nids)] = :unset
    @test_throws ArgumentError validate_field_line_terminations!(RP2; strict = false)
end

@testitem "my_interpolation clamps out-of-domain queries instead of throwing" setup = [FLFFields] begin
    using RAPID2D: my_interpolation

    # The RK4 stepper evaluates the field up to one step beyond the current position
    # BEFORE the wall check can stop the trace, so a trace standing on the last grid
    # node queries just outside the domain. Clamping is load-bearing: with the default
    # NoExtrap this was a DomainError that killed whole runs.
    n = (length(R1D), length(Z1D))
    data = [r + 10z for r in R1D, z in Z1D]
    for method in (:linear, :cubic)
        itp = my_interpolation(R1D, Z1D, data; method)
        @test itp(last(R1D) + 0.5, 0.0) == itp(last(R1D), 0.0)
        @test itp(first(R1D) - 0.5, 0.0) == itp(first(R1D), 0.0)
        @test itp(1.5, last(Z1D) + 0.3) == itp(1.5, last(Z1D))
    end
end
