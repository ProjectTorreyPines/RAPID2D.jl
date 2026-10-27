# Field-line following (FLF) on the grid: the line through every node inside the wall, traced
# both ways along B_pol to the wall or once around, gives the node its connection length
# Lc = ∫ (B/B_pol) dl and poloidal length, and the closed lines their nodes. Any tracer does it;
# a run uses `flags.field_line_tracer`.

"""
    trace_single_field_line(R0, Z0, direction, interp_BR, interp_BZ, interp_Bphi,
                            step_size, max_steps, max_Lpol, wall_checker;
                            detect_closure = true, closure_tolerance = 1e-6) -> SingleTraceResult

One RK4 trace (see [`RK4Tracer`](@ref)) from (R0, Z0) along `direction` (±1) B_pol, in steps of
`step_size`, with the lengths and termination it publishes. `wall_checker(R, Z)` is `true`
inside the wall.
"""
function trace_single_field_line(
        R0::FT, Z0::FT, direction::Int,
        interp_BR, interp_BZ, interp_Bphi,
        step_size::FT, max_steps::Int, max_Lpol::FT,
        wall_checker;
        detect_closure::Bool = true,
        closure_tolerance::FT = FT(1.0e-6)
    ) where {FT <: AbstractFloat}
    lengths = StepLengths{FT}()
    e = rk4_trace!(
        lengths, R0, Z0, direction, interp_BR, interp_BZ, interp_Bphi, step_size, max_steps, max_Lpol, wall_checker;
        detect_closure, closure_tolerance,
    )
    return single_trace_result(lengths, e)
end

# A trace's lengths and ending as a `SingleTraceResult`.
function single_trace_result(lengths, e)
    L = published_lengths(lengths, e.termination, e.R)
    return SingleTraceResult{typeof(L.Lpol)}(;
        L.Lpol, L.Lc, L.min_Bpol, steps = max(lengths.n - 1, 0), e.termination,
        is_closed = e.termination === :closed, hit_wall = e.termination === :wall, final_R = e.R, final_Z = e.Z,
    )
end

"""
    _finalize_total_lengths!(flf) -> flf

Totals are plain sums of the per-direction lengths, except where BOTH terminations are
`:closed`: on a closed surface the two directions retraced one poloidal circuit, and the total
is their mean, the circuit's connection length (≈ 2πqR, what `wall_step_ceiling` reads: a ring's
farthest point is half of it away).
The gate must be the termination pair, not `is_closed`: that flag is set by *either*
direction, and a half-closed pair holds two different real measurements — collapsing
there overwrote the backward one, and `Lpol_tot` feeds `L_mixing`. Per-direction `Lc`
is never touched here.
"""
function _finalize_total_lengths!(flf::FieldLineFollowingResult)
    @. flf.Lpol_tot = flf.Lpol_forward + flf.Lpol_backward
    @. flf.Lc_tot = flf.Lc_forward + flf.Lc_backward

    both_closed = (flf.termination_forward .=== :closed) .&
        (flf.termination_backward .=== :closed)
    for arr in (flf.Lpol_backward, flf.Lpol_tot)
        @. arr[both_closed] = flf.Lpol_forward[both_closed]
    end
    @. flf.Lc_tot[both_closed] = (flf.Lc_forward[both_closed] + flf.Lc_backward[both_closed]) / 2
    return flf
end

"""
    field_line_analysis!(flf, g, tracer; max_length = default_max_length(g)) -> flf

Field-line following on every node of the geometry `g`: the line through each node traced
forward and backward along B_pol by `tracer` (backward is skipped when forward closed), and the
lengths, terminations and closed nodes it publishes stored in `flf`. A node the tracer cannot
follow (`:degenerate`, marching squares next to an O-point) is traced by its fallback,
[`CubicTracer`](@ref). A contour tracer's line that closes one way is closed both ways.
"""
function field_line_analysis!(
        flf::FieldLineFollowingResult{FT}, g::FieldLineGeometry, tracer::FieldLineTracer;
        max_length::Real = default_max_length(g),
    ) where {FT <: AbstractFloat}
    NR, NZ = size(g.ψ)
    flf.max_Lpol = max_length
    flf.max_step = 2 * step_budget(tracer, g, max_length)
    empty!(flf.closed_surface_nids)
    fill!(flf.is_closed, false)
    # Back to "no trace has run" rather than to a plausible status, so a node the loop below
    # fails to reach cannot pass a topology check on a stale answer.
    fill!(flf.termination_forward, :unset)
    fill!(flf.termination_backward, :unset)
    Lf_visitor, Lb_visitor = path_lengths(tracer, FT), path_lengths(tracer, FT)
    for i in 1:NR, j in 1:NZ
        fwd, ef, t = node_trace!(reset!(Lf_visitor), tracer, g, i, j, 1, max_length)
        Lf = published_lengths(fwd, ef.termination, ef.R)
        if ef.termination === :closed
            # a closed forward trace already covers the circuit
            bwd, eb, Lb = fwd, ef, Lf
        else
            bwd, eb, t = node_trace!(reset!(Lb_visitor), t, g, i, j, -1, max_length)
            Lb = published_lengths(bwd, eb.termination, eb.R)
            # a contour that came back to its node is closed, however the other way ended
            eb.termination === :closed && closes_by_return(t) && ((fwd, ef, Lf) = (bwd, eb, Lb))
        end
        flf.Lpol_forward[i, j], flf.Lpol_backward[i, j] = Lf.Lpol, Lb.Lpol
        flf.Lc_forward[i, j], flf.Lc_backward[i, j] = Lf.Lc, Lb.Lc
        flf.min_Bpol[i, j] = min(Lf.min_Bpol, Lb.min_Bpol)
        flf.step[i, j] = max(fwd.n - 1, 0) + max(bwd.n - 1, 0)
        flf.termination_forward[i, j], flf.termination_backward[i, j] = ef.termination, eb.termination
        if ef.termination === :closed || eb.termination === :closed
            flf.is_closed[i, j] = true
            push!(flf.closed_surface_nids, (j - 1) * NR + i)
        end
    end
    return _finalize_total_lengths!(flf)
end

# One direction from node (i, j) into the lengths visitor `L`: the visitor that measured it, the
# ending, and the tracer that did it (the fallback, with a fresh visitor, when `tracer` could not
# follow the node's level).
function node_trace!(L, tracer, g, i, j, sgn, max_length)
    e = trace_from_node!(L, tracer, g, i, j, sgn; max_length)
    e.termination === :degenerate || return L, e, tracer
    fallback = fallback_tracer(tracer)
    return node_trace!(path_lengths(fallback, typeof(L.Lpol)), fallback, g, i, j, sgn, max_length)
end

"""
    flf_analysis_field_lines_rz_plane!(flf, R1D, Z1D, BR, BZ, Bϕ, cell_state; dR, dZ, out_wall_idx) -> flf
    flf_analysis_field_lines_rz_plane(R1D, Z1D, BR, BZ, Bϕ, cell_state; kwargs...) -> flf

[`field_line_analysis!`](@ref) with [`RK4Tracer`](@ref) on nodal fields alone (no ψ): `cell_state`
marks the in-wall cells, and `min_Bpol` is `NaN` at `out_wall_idx`.
"""
function flf_analysis_field_lines_rz_plane!(
        flf::FieldLineFollowingResult{FT},
        R1D::Vector{FT}, Z1D::Vector{FT}, BR::Matrix{FT}, BZ::Matrix{FT}, Bϕ::Matrix{FT},
        cell_state::AbstractMatrix{Bool};
        dR::Union{FT, Nothing} = nothing, dZ::Union{FT, Nothing} = nothing,
        out_wall_idx::Union{Vector{Int}, Nothing} = nothing,
    ) where {FT <: AbstractFloat}
    dR === nothing && (dR = length(R1D) > 1 ? R1D[2] - R1D[1] : one(FT))
    dZ === nothing && (dZ = length(Z1D) > 1 ? Z1D[2] - Z1D[1] : one(FT))
    B = (BR = my_interpolation(R1D, Z1D, BR), BZ = my_interpolation(R1D, Z1D, BZ), Bϕ = my_interpolation(R1D, Z1D, Bϕ))
    # RK4 reads the grid, the wall and these interpolants, not ψ
    g = FieldLineGeometry(R1D, Z1D, dR, dZ, zeros(FT, size(BR)), zero(FT), nothing, Bϕ, cell_state, B)
    field_line_analysis!(flf, g, RK4Tracer())
    out_wall_idx !== nothing && (flf.min_Bpol[out_wall_idx] .= FT(NaN))
    return flf
end

function flf_analysis_field_lines_rz_plane(
        R1D::Vector{FT}, Z1D::Vector{FT}, BR::Matrix{FT}, BZ::Matrix{FT}, Bϕ::Matrix{FT},
        cell_state::AbstractMatrix{Bool};
        dR::Union{FT, Nothing} = nothing, dZ::Union{FT, Nothing} = nothing,
        out_wall_idx::Union{Vector{Int}, Nothing} = nothing,
    ) where {FT <: AbstractFloat}
    flf = FieldLineFollowingResult{FT}(; dims_RZ = (length(R1D), length(Z1D)))
    return flf_analysis_field_lines_rz_plane!(flf, R1D, Z1D, BR, BZ, Bϕ, cell_state; dR, dZ, out_wall_idx)
end

# The analysis of a run's field by its own tracer, into a new result.
function flf_analysis_field_lines_rz_plane(RP::RAPID{FT}) where {FT}
    flf = FieldLineFollowingResult{FT}(RP.G.NR, RP.G.NZ)
    return run_field_line_analysis!(flf, RP)
end

function run_field_line_analysis!(flf::FieldLineFollowingResult{FT}, RP::RAPID{FT}) where {FT}
    tracer = RP.flags.field_line_tracer
    field_line_analysis!(flf, FieldLineGeometry(RP; nodal_B = needs_nodal_B(tracer)), tracer)
    flf.min_Bpol[RP.G.nodes.out_wall_nids] .= FT(NaN)
    return flf
end

"""
    validate_field_line_terminations!(RP; strict = true) -> RP.flf

Check that every **in-wall** node has a topology transport can act on.

`:trace_limit` means the distance is UNKNOWN (`Lc = Inf`), and silently disabling the
geometric ceiling on a failed measurement is the failure it exists to prevent:

- `strict = true` (setup): throw — a run must not start from an unmeasured geometry.
- `strict = false` (periodic step-loop refresh): warn once and substitute the
  conservative `Lpol_partial · Btot/Bpol` — at least that much was walked, so the
  ceiling gets tighter than truth, never looser. Termination stays `:trace_limit`.

`:unset` (an untraced node) is a code bug and throws in both modes. `:null` and a
half-closed pair only warn — both publish finite lengths, so the ceiling stays active.
Out-of-wall nodes are not validated: a trace starting outside returns `:wall` at zero
distance before its first step.
"""
function validate_field_line_terminations!(RP::RAPID; strict::Bool = true)
    flf = RP.flf
    tf, tb = flf.termination_forward, flf.termination_backward
    nids = RP.G.nodes.in_wall_nids
    # `R2D`/`Z2D` accept a linear node id directly — no hand-rolled index arithmetic.
    coords(nid) = (RP.G.R2D[nid], RP.G.Z2D[nid])

    unmeasured = [n for n in nids if tf[n] === :unset || tb[n] === :unset]
    isempty(unmeasured) || throw(
        ArgumentError(
            "field-line following left $(length(unmeasured)) of $(length(nids)) in-wall " *
                "nodes untraced, e.g. at (R, Z) = $(coords(first(unmeasured))). " *
                "Transport cannot read a length that was never measured."
        )
    )

    budget = [n for n in nids if tf[n] === :trace_limit || tb[n] === :trace_limit]
    if !isempty(budget)
        examples = join((string(coords(n)) for n in first(budget, 3)), ", ")
        strict && throw(
            ArgumentError(
                "field-line tracing ran out of budget at $(length(budget)) of " *
                    "$(length(nids)) in-wall nodes, e.g. (R, Z) = $examples. " *
                    "The distance to the wall there is UNKNOWN, not infinite, so the " *
                    "geometric transport ceiling cannot be switched off on its account. " *
                    "The budget was max_Lpol = $(flf.max_Lpol) m over at most " *
                    "$(flf.max_step) steps, derived as 3× the domain diagonal; raise it " *
                    "(or shrink the domain) if these field lines are genuinely that long."
            )
        )
        # Non-strict (step-loop refresh): substitute the partial parallel length
        # actually walked, `Lpol_partial · Btot/Bpol` — the true distance is at least
        # that, so the ceiling gets tighter than truth, never looser. Termination stays
        # `:trace_limit`: this is a fallback length, not a reclassification.
        F = RP.fields
        for n in budget
            ratio = F.Btot[n] / hypot(F.BR[n], F.BZ[n])
            for (t, Lc, Lpol) in (
                    (tf, flf.Lc_forward, flf.Lpol_forward),
                    (tb, flf.Lc_backward, flf.Lpol_backward),
                )
                t[n] === :trace_limit || continue
                fb = Lpol[n] * ratio
                # Belt: a degenerate node must not re-inject Inf.
                Lc[n] = isfinite(fb) ? fb : oftype(Lc[n], 2π) * RP.G.R2D[n]
            end
            flf.Lc_tot[n] = flf.Lc_forward[n] + flf.Lc_backward[n]
        end
        # `maxlog`: the same struggling trace would otherwise re-warn every FLF_nstep.
        @warn(
            "field-line tracing ran out of budget; substituted the partial walked " *
                "length as a conservative wall distance (ceiling tighter than truth, " *
                "never looser). Raise max_Lpol if these lines are genuinely that long.",
            nodes = "$(length(budget))/$(length(nids))",
            example_RZ = coords(first(budget)),
            max_Lpol = flf.max_Lpol,
            maxlog = 1
        )
    end

    # Warnings from here down: each leaves the ceiling off at the affected nodes, which
    # is the behaviour that predates this contract, so none of them is a regression.
    # `maxlog = 1`: static geometry would re-warn identically on every FLF refresh.
    nulls = [n for n in nids if tf[n] === :null || tb[n] === :null]
    isempty(nulls) || @warn(
        "field-line tracing hit a Bpol null; the field is purely toroidal there, so the " *
            "line is treated as a closed circle of circumference 2πR",
        nodes = "$(length(nulls))/$(length(nids))",
        example_RZ = coords(first(nulls)),
        maxlog = 1
    )

    half_closed = [n for n in nids if (tf[n] === :closed) ⊻ (tb[n] === :closed)]
    isempty(half_closed) || @warn(
        "closure detected in one direction only — likely a strongly curved open line " *
            "(the detector accumulates turning angle, not return-to-start). Both " *
            "directions still published finite lengths, so the ceiling stays active.",
        nodes = "$(length(half_closed))/$(length(nids))",
        example_RZ = coords(first(half_closed)),
        maxlog = 1
    )

    return flf
end

function flf_analysis_field_lines_rz_plane!(RP::RAPID; strict::Bool = true)
    run_field_line_analysis!(RP.flf, RP)
    # Validated HERE so the setup call and every periodic refresh share one gate; the
    # lower-level array API stays unvalidated and separately testable. `strict` picks
    # the failure policy — throw at setup, warn-and-fallback in the step loop.
    return validate_field_line_terminations!(RP; strict)
end
