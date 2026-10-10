"""
Field Line Following (FLF) Analysis in RZ Plane

This module provides functions for magnetic field line tracing in cylindrical (RZ) coordinates,
including forward/backward field line following, closed field line detection, and connection
length calculations.
"""

using FastInterpolations
using LinearAlgebra
using RAPID2D

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

Totals are plain sums of the per-direction lengths, collapsed onto one direction only
where BOTH terminations are `:closed` — the two directions then retraced one circuit.
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
    @. flf.Lc_tot[both_closed] = flf.Lc_forward[both_closed]
    return flf
end

"""
    flf_analysis_field_lines_rz_plane(R1D, Z1D, BR, BZ, Bϕ, cell_state;
                                         dR=nothing, dZ=nothing,
                                         out_wall_idx=nothing,
                                         )

Modular field line following analysis using Julia best practices.

This version replaces the complex vectorized approach with a clean,
modular design that traces each field line individually.

# Features
- Clean, readable code following Julia best practices
- Type-stable implementation for better performance
- Easy to optimize with @inbounds and other performance annotations
- Built-in parallelization support
- Modular design for easy testing and debugging

# Arguments
- `R1D::Vector{FT}`: 1D radial coordinate array
- `Z1D::Vector{FT}`: 1D vertical coordinate array
- `BR::Matrix{FT}`: Radial magnetic field component
- `BZ::Matrix{FT}`: Vertical magnetic field component
- `Bϕ::Matrix{FT}`: Toroidal magnetic field component
- `cell_state::Vector{Bool}`: Cell state array for wall boundary checking
- `dR::Union{FT,Nothing}`: Radial grid spacing (optional)
- `dZ::Union{FT,Nothing}`: Vertical grid spacing (optional)
- `out_wall_idx::Union{Vector{Int},Nothing}`: Indices outside wall (optional)

# Returns
- `flf::FieldLineFollowingResult`: Field line following results
- `fmap2d::Union{FieldMapResult, Nothing}`: 2D field mapping results (if requested)

# Example
```julia
# Use the new modular version
flf_result, _ = flf_analysis_field_lines_rz_plane(
    R1D, Z1D, BR, BZ, Bϕ, cell_state, parallel=true
)
```
"""
function flf_analysis_field_lines_rz_plane!(
        flf::FieldLineFollowingResult{FT},
        R1D::Vector{FT}, Z1D::Vector{FT}, BR::Matrix{FT}, BZ::Matrix{FT}, Bϕ::Matrix{FT},
        cell_state::AbstractMatrix{Bool};
        dR::Union{FT, Nothing} = nothing, dZ::Union{FT, Nothing} = nothing,
        out_wall_idx::Union{Vector{Int}, Nothing} = nothing,
    ) where {FT <: AbstractFloat}

    # @assert size(flf.Lc_tot) == (length(R1D), length(Z1D)) "FieldLineFollowingResult size mismatch"

    # Compute grid spacing if not provided
    if dR === nothing
        dR = length(R1D) > 1 ? R1D[2] - R1D[1] : one(FT)
    end
    if dZ === nothing
        dZ = length(Z1D) > 1 ? Z1D[2] - Z1D[1] : one(FT)
    end

    NR, NZ = length(R1D), length(Z1D)

    # Grid parameters
    Rmin = R1D[1]
    Zmin = Z1D[1]
    # NZ = length(Z1D)
    inv_dR = one(FT) / dR
    inv_dZ = one(FT) / dZ

    # Set up interpolants for magnetic field components
    interp_BR = my_interpolation(R1D, Z1D, BR)
    interp_BZ = my_interpolation(R1D, Z1D, BZ)
    interp_Bϕ = my_interpolation(R1D, Z1D, Bϕ)

    # Integration parameters
    step_size = FT(0.5) * min(abs(dR), abs(dZ))
    max_Lpol = FT(3) * sqrt((maximum(R1D) - minimum(R1D))^2 + (maximum(Z1D) - minimum(Z1D))^2)
    max_step_per_direction = floor(Int, max_Lpol / step_size)

    flf.max_Lpol = max_Lpol
    flf.max_step = 2 * max_step_per_direction

    # Create wall checker function
    wall_checker = (R, Z) -> is_in_wall_by_cell_state(
        R, Z, Rmin, Zmin, NR, inv_dR, inv_dZ, cell_state
    )


    empty!(flf.closed_surface_nids)  # Clear previous results
    fill!(flf.is_closed, false)  # Reset closed flags
    # Back to "no trace has run" rather than to a plausible status, so a node the loop
    # below fails to reach cannot pass a topology check on a stale answer.
    fill!(flf.termination_forward, :unset)
    fill!(flf.termination_backward, :unset)

    @inbounds for i in 1:NR, j in 1:NZ
        R0, Z0 = R1D[i], Z1D[j]

        # Forward tracing
        forward_result = trace_single_field_line(
            R0, Z0, 1, interp_BR, interp_BZ, interp_Bϕ,
            step_size, max_step_per_direction, max_Lpol, wall_checker
        )

        # Backward tracing (skip if already closed)
        backward_result = if forward_result.is_closed
            # The one construction site that never runs the tracer: a closed forward
            # trace already covers the circuit. Keyword form — positional args silently
            # reshuffle when a field is added.
            SingleTraceResult{FT}(;
                Lpol = forward_result.Lpol, Lc = forward_result.Lc,
                min_Bpol = forward_result.min_Bpol, steps = forward_result.steps,
                termination = :closed, is_closed = true, hit_wall = false,
                final_R = R0, final_Z = Z0
            )
        else
            trace_single_field_line(
                R0, Z0, -1, interp_BR, interp_BZ, interp_Bϕ,
                step_size, max_step_per_direction, max_Lpol, wall_checker
            )
        end

        # Store results
        flf.Lpol_forward[i, j] = forward_result.Lpol
        flf.Lpol_backward[i, j] = backward_result.Lpol
        flf.Lc_forward[i, j] = forward_result.Lc
        flf.Lc_backward[i, j] = backward_result.Lc
        flf.min_Bpol[i, j] = min(forward_result.min_Bpol, backward_result.min_Bpol)
        flf.step[i, j] = forward_result.steps + backward_result.steps
        flf.termination_forward[i, j] = forward_result.termination
        flf.termination_backward[i, j] = backward_result.termination

        if forward_result.is_closed || backward_result.is_closed
            flf.is_closed[i, j] = true
            push!(flf.closed_surface_nids, (j - 1) * NR + i)  # Store linear index of closed field line
        end
    end

    _finalize_total_lengths!(flf)

    # Set NaN values for points outside wall
    if out_wall_idx !== nothing
        @. flf.min_Bpol[out_wall_idx] = FT(NaN)
    end

    return flf
end

function flf_analysis_field_lines_rz_plane(
        R1D::Vector{FT}, Z1D::Vector{FT}, BR::Matrix{FT}, BZ::Matrix{FT}, Bϕ::Matrix{FT},
        cell_state::AbstractMatrix{Bool};
        dR::Union{FT, Nothing} = nothing, dZ::Union{FT, Nothing} = nothing,
        out_wall_idx::Union{Vector{Int}, Nothing} = nothing,
    ) where {FT <: AbstractFloat}

    flf = FieldLineFollowingResult{FT}(; dims_RZ = (length(R1D), length(Z1D)))
    return flf_analysis_field_lines_rz_plane!(
        flf, R1D, Z1D, BR, BZ, Bϕ, cell_state;
        dR, dZ, out_wall_idx
    )
end

# Convenience dispatch for RAPID object
function flf_analysis_field_lines_rz_plane(RP::RAPID)
    return flf_analysis_field_lines_rz_plane(
        RP.G.R1D, RP.G.Z1D, RP.fields.BR, RP.fields.BZ, RP.fields.Bϕ,
        RP.G.cell_state .>= 0; # Use cell_state as boolean mask
        dR = RP.G.dR, dZ = RP.G.dZ,
        out_wall_idx = RP.G.nodes.out_wall_nids
    )
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
    flf_analysis_field_lines_rz_plane!(
        RP.flf,
        RP.G.R1D, RP.G.Z1D, RP.fields.BR, RP.fields.BZ, RP.fields.Bϕ,
        RP.G.cell_state .>= 0; # Use cell_state as boolean mask
        dR = RP.G.dR, dZ = RP.G.dZ,
        out_wall_idx = RP.G.nodes.out_wall_nids
    )
    # Validated HERE so the setup call and every periodic refresh share one gate; the
    # lower-level array API stays unvalidated and separately testable. `strict` picks
    # the failure policy — throw at setup, warn-and-fallback in the step loop.
    return validate_field_line_terminations!(RP; strict)
end
