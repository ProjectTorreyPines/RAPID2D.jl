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
    advance_step_along_b_rz_plane(dl, R, Z, interp_BR, interp_BZ)

Advance one step along magnetic field line using 4th-order Runge-Kutta integration.

This function implements the RK4 algorithm for integrating field lines in the poloidal
plane, following the magnetic field direction with step size `dl`.

# Arguments
- `dl::FT`: Step size (positive for forward, negative for backward)
- `R::FT`: Current radial position
- `Z::FT`: Current vertical position
- `interp_BR`: Radial magnetic field interpolation function
- `interp_BZ`: Vertical magnetic field interpolation function

# Returns
- `next_R::FT`: New radial position
- `next_Z::FT`: New vertical position

# Algorithm
Uses classical RK4 integration:
1. k1 = f(t, y)
2. k2 = f(t + h/2, y + k1*h/2)
3. k3 = f(t + h/2, y + k2*h/2)
4. k4 = f(t + h, y + k3*h)
5. y_new = y + h*(k1 + 2*k2 + 2*k3 + k4) / 6

Where f(t,y) = B(R,Z)/|B_pol(R,Z)| is the unit field direction.
"""
function advance_step_along_b_rz_plane(
        dl::FT,
        R::FT,
        Z::FT,
        interp_BR::FastInterpolations.AbstractInterpolantND,
        interp_BZ::FastInterpolations.AbstractInterpolantND
    ) where {FT <: AbstractFloat}
    half = FT(0.5)
    two_FT = FT(2.0)

    # RK4 Step 1
    BR1 = interp_BR(R, Z)
    BZ1 = interp_BZ(R, Z)
    Bpol1 = sqrt(BR1^two_FT + BZ1^two_FT)
    R_RK1 = dl * BR1 / Bpol1
    Z_RK1 = dl * BZ1 / Bpol1

    # RK4 Step 2
    R2 = R + half * R_RK1
    Z2 = Z + half * Z_RK1
    BR2 = interp_BR(R2, Z2)
    BZ2 = interp_BZ(R2, Z2)
    Bpol2 = sqrt(BR2^two_FT + BZ2^two_FT)
    R_RK2 = dl * BR2 / Bpol2
    Z_RK2 = dl * BZ2 / Bpol2

    # RK4 Step 3
    R3 = R + half * R_RK2
    Z3 = Z + half * Z_RK2
    BR3 = interp_BR(R3, Z3)
    BZ3 = interp_BZ(R3, Z3)
    Bpol3 = sqrt(BR3^two_FT + BZ3^two_FT)
    R_RK3 = dl * BR3 / Bpol3
    Z_RK3 = dl * BZ3 / Bpol3

    # RK4 Step 4
    R4 = R + R_RK3
    Z4 = Z + Z_RK3
    BR4 = interp_BR(R4, Z4)
    BZ4 = interp_BZ(R4, Z4)
    Bpol4 = sqrt(BR4^two_FT + BZ4^two_FT)
    R_RK4 = dl * BR4 / Bpol4
    Z_RK4 = dl * BZ4 / Bpol4

    # Combine RK4 steps
    next_R = R + (R_RK1 + two_FT * R_RK2 + two_FT * R_RK3 + R_RK4) / FT(6.0)
    next_Z = Z + (Z_RK1 + two_FT * Z_RK2 + two_FT * Z_RK3 + Z_RK4) / FT(6.0)

    return next_R, next_Z
end

"""
    is_in_wall_by_cell_state(R, Z, Rmin, Zmin, NZ, inv_dR, inv_dZ, cell_state)

Check if point (R,Z) is inside the computational domain (not in wall).

Uses cell state array to determine if a point is inside the valid computational
region or hits a wall boundary.

# Arguments
- `R::FT`: Radial coordinate
- `Z::FT`: Vertical coordinate
- `Rmin::FT`: Minimum radial coordinate
- `Zmin::FT`: Minimum vertical coordinate
- `NZ::Int`: Number of vertical grid points
- `inv_dR::FT`: Inverse radial grid spacing
- `inv_dZ::FT`: Inverse vertical grid spacing
- `cell_state::Vector{Bool}`: Cell state array (true = valid, false = wall)

# Returns
- `state::Bool`: true if point is valid (not in wall), false if in wall
"""
function is_in_wall_by_cell_state(
        R::FT, Z::FT, Rmin::FT, Zmin::FT, NR::Int,
        inv_dR::FT, inv_dZ::FT, cell_state::AbstractMatrix{Bool}
    ) where {FT <: AbstractFloat}

    # Convert coordinates to grid indices
    Rid = floor(Int, (R - Rmin) * inv_dR) + 1
    Zid = floor(Int, (Z - Zmin) * inv_dZ) + 1

    # Bound R FIRST: the linear index alone folds `Rid = NR+1` onto the valid index of
    # `(1, Zid+1)`, so a rightward escape used to read as "still inside".
    if Rid < 1 || Rid > NR
        return false  # Outside domain = wall
    end

    # With R bounded, the linear check below is exactly a bound on Zid.
    nid = (Zid - 1) * NR + Rid
    if nid < 1 || nid > length(cell_state)
        return false  # Outside domain = wall
    end

    return cell_state[nid]
end

"""
    trace_single_field_line(R0, Z0, direction, interp_BR, interp_BZ, interp_Bphi,
                           step_size, max_steps, max_Lpol, wall_checker;
                           detect_closure=true, closure_tolerance=1e-6)

Trace a single magnetic field line from starting point (R0, Z0).

This function follows Julia best practices with:
- Type-stable implementation
- Minimal memory allocation
- Clear control flow
- Easy to optimize with @inbounds

# Arguments
- `R0::FT`: Starting radial coordinate
- `Z0::FT`: Starting vertical coordinate
- `direction::Int`: +1 for forward, -1 for backward
- `interp_BR`: Radial magnetic field interpolation function
- `interp_BZ`: Vertical magnetic field interpolation function
- `interp_Bphi`: Toroidal magnetic field interpolation function
- `step_size::FT`: Integration step size
- `max_steps::Int`: Maximum number of steps
- `max_Lpol::FT`: Maximum poloidal length
- `wall_checker`: Function to check if point hits wall
- `detect_closure::Bool`: Whether to detect closed field lines
- `closure_tolerance::FT`: Tolerance for closure detection

# Returns
- `SingleTraceResult`: Complete tracing results
"""
function trace_single_field_line(
        R0::FT, Z0::FT, direction::Int,
        interp_BR, interp_BZ, interp_Bphi,
        step_size::FT, max_steps::Int, max_Lpol::FT,
        wall_checker;
        detect_closure::Bool = true,
        closure_tolerance::FT = FT(1.0e-6)
    ) where {FT <: AbstractFloat}

    # Initialize current position
    R_current = R0
    Z_current = Z0

    # Initialize accumulated quantities
    Lpol = zero(FT)
    Lc = zero(FT)
    min_Bpol = FT(Inf)
    steps = 0

    # For closure detection
    total_angle = zero(FT)
    prev_R, prev_Z = R0, Z0

    # Integration step size with direction
    dl = direction * step_size


    # Check if starting point is a null point (Bpol = 0)
    BR = interp_BR(R_current, Z_current)
    BZ = interp_BZ(R_current, Z_current)
    Bpol = sqrt(BR^2 + BZ^2)

    if Bpol == 0
        # `Bpol = 0` means purely toroidal: the line is a closed circle, `Lc = 2πR`.
        # `Lpol` keeps its existing `Inf` — its honest value here is a separate question,
        # since `Lpol_tot` feeds `L_mixing`
        # (internal/docs/src/notes/TODO/L-mixing-serves-two-lengths.md). `min_Bpol` is
        # zero by the same token that makes this `:null`, not the `Inf` seed.
        return SingleTraceResult{FT}(;
            Lpol = FT(Inf), Lc = FT(2π) * R_current, min_Bpol = zero(FT), steps,
            termination = :null,
            is_closed = false, hit_wall = false, final_R = R_current, final_Z = Z_current
        )
    end

    # Main integration loop
    for step in 1:max_steps
        # Check wall boundary
        if !wall_checker(R_current, Z_current)
            # THE ONLY EXIT THAT PUBLISHES A FINITE `Lc`.
            return SingleTraceResult{FT}(;
                Lpol, Lc, min_Bpol, steps, termination = :wall,
                is_closed = false, hit_wall = true, final_R = R_current, final_Z = Z_current
            )
        end

        # Store previous position for step calculation
        R_prev = R_current
        Z_prev = Z_current

        # Advance one step using RK4
        R_current, Z_current = advance_step_along_b_rz_plane(
            dl, R_current, Z_current, interp_BR, interp_BZ
        )

        # The RK4 stages divide by Bpol at their lookahead points, so a step poking
        # into a Bpol = 0 region comes back NaN before the committed-position check
        # below can see it. Still a null termination, at the last position actually
        # reached; unguarded, the NaN misread as a wall hit (NaN comparisons are
        # false) or threw in the cell-state checker (`floor(Int, NaN)`).
        if !(isfinite(R_current) && isfinite(Z_current))
            return SingleTraceResult{FT}(;
                Lpol = FT(Inf), Lc = Lc + FT(2π) * R_prev, min_Bpol = zero(FT), steps,
                termination = :null,
                is_closed = false, hit_wall = false, final_R = R_prev, final_Z = Z_prev
            )
        end

        # Calculate magnetic field components at new position
        BR = interp_BR(R_current, Z_current)
        BZ = interp_BZ(R_current, Z_current)
        Bphi = interp_Bphi(R_current, Z_current)

        Bpol = sqrt(BR^2 + BZ^2)

        if Bpol == 0
            # Same reading as the starting null above, with the distance already walked
            # to get here: the line runs `Lc` to this point and then closes toroidally.
            return SingleTraceResult{FT}(;
                Lpol = FT(Inf), Lc = Lc + FT(2π) * R_current, min_Bpol = zero(FT), steps,
                termination = :null,
                is_closed = false, hit_wall = false, final_R = R_current, final_Z = Z_current
            )
        end

        Btot = sqrt(Bpol^2 + Bphi^2)

        # Calculate step lengths
        dl_pol = sqrt((R_current - R_prev)^2 + (Z_current - Z_prev)^2)
        dl_tot = dl_pol * Btot / Bpol

        # Update accumulated quantities
        Lpol += dl_pol
        Lc += dl_tot
        min_Bpol = min(min_Bpol, Bpol)
        steps += 1

        # Check maximum poloidal length
        if Lpol > max_Lpol
            # `Lc = Inf` means UNKNOWN, not "no wall": the trace was still going when
            # the budget ran out. `Lpol` keeps the distance actually travelled.
            return SingleTraceResult{FT}(;
                Lpol, Lc = FT(Inf), min_Bpol, steps, termination = :trace_limit,
                is_closed = false, hit_wall = false, final_R = R_current, final_Z = Z_current
            )
        end

        # Closure detection using angle accumulation
        if detect_closure && step > 2
            # Calculate angle between consecutive displacement vectors
            v1_R, v1_Z = R_prev - prev_R, Z_prev - prev_Z
            v2_R, v2_Z = R_current - R_prev, Z_current - Z_prev

            # Cross and dot products for angle calculation
            cross_prod = v1_R * v2_Z - v1_Z * v2_R
            dot_prod = v1_R * v2_R + v1_Z * v2_Z

            if dot_prod != 0 || cross_prod != 0
                angle = atan(cross_prod, dot_prod)
                total_angle += angle

                # Check for 360° circulation
                if abs(total_angle) >= 2π - closure_tolerance
                    # `Lc` is the CIRCUIT length, not `Inf`: the geometry repeats
                    # after one circuit, so a longer step reaches nowhere new — a
                    # measured bound, same standing as a wall distance.
                    return SingleTraceResult{FT}(;
                        Lpol, Lc, min_Bpol, steps, termination = :closed,
                        is_closed = true, hit_wall = false, final_R = R_current, final_Z = Z_current
                    )
                end
            end

            prev_R, prev_Z = R_prev, Z_prev
        end
    end


    # Reached maximum steps — the same physical condition as `Lpol > max_Lpol` above
    # (`max_step_per_direction = floor(max_Lpol/step_size)` makes them trip together),
    # so it carries the same status rather than one of its own.
    return SingleTraceResult{FT}(;
        Lpol, Lc = FT(Inf), min_Bpol, steps, termination = :trace_limit,
        is_closed = false, hit_wall = false, final_R = R_current, final_Z = Z_current
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

# Helper function to create 2D interpolation that matches MATLAB's griddedInterpolant behavior
function my_interpolation(R1D::Vector{FT}, Z1D::Vector{FT}, data_2d::Matrix{FT}; method::Symbol = :cubic) where {FT <: AbstractFloat}
    @assert method in (:nearst, :linear, :cubic) "Invalid interpolation method: $method"

    r1d = range(R1D[1], stop = R1D[end], length = length(R1D))
    z1d = range(Z1D[1], stop = Z1D[end], length = length(Z1D))

    # `ClampExtrap`, not the default `NoExtrap`: RK4 evaluates up to one step BEYOND
    # the current position before the next wall check can stop the trace, so a trace on
    # the last grid node queries just outside the domain — which `NoExtrap` answered
    # with a run-killing DomainError. Outside the grid there is no field data, so the
    # edge IS a wall; the clamped step is discarded by the very next wall check, and
    # flat clamping cannot diverge the way extending the cubic would.
    extrap = ClampExtrap()
    if method == :nearst
        itp = constant_interp((r1d, z1d), data_2d; extrap)
    elseif method == :linear
        itp = linear_interp((r1d, z1d), data_2d; extrap)
    elseif method == :cubic
        itp = cubic_interp((r1d, z1d), data_2d; extrap)
    end
    return itp
end
