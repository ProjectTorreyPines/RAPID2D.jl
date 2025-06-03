"""
Field Line Following (FLF) Analysis in RZ Plane

This module provides functions for magnetic field line tracing in cylindrical (RZ) coordinates,
including forward/backward field line following, closed field line detection, and connection
length calculations.
"""

using Interpolations
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
    interp_BR::Interpolations.GriddedInterpolation,
    interp_BZ::Interpolations.GriddedInterpolation
) where FT<:AbstractFloat
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
) where FT<:AbstractFloat

    # Convert coordinates to grid indices
    Rid = floor(Int, (R - Rmin) * inv_dR) + 1
    Zid = floor(Int, (Z - Zmin) * inv_dZ) + 1

    # Convert 2D indices to linear index
    nid = (Zid - 1) * NR + Rid

    # Check bounds and return cell state
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
    closure_tolerance::FT = FT(1e-6)
) where FT <: AbstractFloat

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
        return SingleTraceResult{FT}(;
            Lpol=FT(Inf), Lc=FT(Inf), min_Bpol, steps,
            is_closed=false, hit_wall=false, final_R = R_current, final_Z = Z_current
        )
    end

    # Main integration loop
    for step in 1:max_steps
        # Check wall boundary
        if !wall_checker(R_current, Z_current)
            return SingleTraceResult{FT}(;
                Lpol, Lc, min_Bpol, steps,
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

        # Calculate magnetic field components at new position
        BR = interp_BR(R_current, Z_current)
        BZ = interp_BZ(R_current, Z_current)
        Bphi = interp_Bphi(R_current, Z_current)

        Bpol = sqrt(BR^2 + BZ^2)

        if Bpol == 0
            return SingleTraceResult{FT}(;
                Lpol=FT(Inf), Lc=FT(Inf), min_Bpol, steps,
                is_closed=false, hit_wall=false, final_R = R_current, final_Z = Z_current
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
            return SingleTraceResult{FT}(;
                Lpol, Lc, min_Bpol, steps,
                is_closed=false, hit_wall=false, final_R = R_current, final_Z = Z_current
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
                    return SingleTraceResult{FT}(;
                        Lpol, Lc, min_Bpol, steps,
                        is_closed=true, hit_wall=false, final_R = R_current, final_Z = Z_current
                    )
                end
            end

            prev_R, prev_Z = R_prev, Z_prev
        end
    end


    # Reached maximum steps
    return SingleTraceResult{FT}(;
        Lpol, Lc, min_Bpol, steps,
        is_closed=false, hit_wall=false, final_R = R_current, final_Z = Z_current
    )
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
    dR::Union{FT,Nothing}=nothing, dZ::Union{FT,Nothing}=nothing,
    out_wall_idx::Union{Vector{Int},Nothing}=nothing,
) where FT<:AbstractFloat

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

    @inbounds for i in 1:NR, j in 1:NZ
        R0, Z0 = R1D[i], Z1D[j]

        # Forward tracing
        forward_result = trace_single_field_line(
            R0, Z0, 1, interp_BR, interp_BZ, interp_Bϕ,
            step_size, max_step_per_direction, max_Lpol, wall_checker
        )

        # Backward tracing (skip if already closed)
        backward_result = if forward_result.is_closed
            SingleTraceResult(
                forward_result.Lpol, forward_result.Lc, forward_result.min_Bpol,
                forward_result.steps, true, false, R0, Z0
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

        if forward_result.is_closed || backward_result.is_closed
            flf.is_closed[i, j] = true
            push!(flf.closed_surface_nids, (j - 1) * NR + i)  # Store linear index of closed field line
        end
    end

    # Calculate total lengths
    @. flf.Lpol_tot = flf.Lpol_forward + flf.Lpol_backward
    @. flf.Lc_tot = flf.Lc_forward + flf.Lc_backward

    # Handle closed field lines (total = forward = backward for closed lines)
    @. flf.Lpol_backward[flf.is_closed] = flf.Lpol_forward[flf.is_closed]
    @. flf.Lc_backward[flf.is_closed] = flf.Lc_forward[flf.is_closed]

    # For closed field lines, total equals one direction
    @. flf.Lpol_tot[flf.is_closed] = flf.Lpol_forward[flf.is_closed]
    @. flf.Lc_tot[flf.is_closed] = flf.Lc_forward[flf.is_closed]

    # Set NaN values for points outside wall
    if out_wall_idx !== nothing
        @. flf.min_Bpol[out_wall_idx] = FT(NaN)
    end

    return flf
end

function flf_analysis_field_lines_rz_plane(
    R1D::Vector{FT}, Z1D::Vector{FT}, BR::Matrix{FT}, BZ::Matrix{FT}, Bϕ::Matrix{FT},
    cell_state::AbstractMatrix{Bool};
    dR::Union{FT,Nothing}=nothing, dZ::Union{FT,Nothing}=nothing,
    out_wall_idx::Union{Vector{Int},Nothing}=nothing,
) where FT<:AbstractFloat

    flf = FieldLineFollowingResult{FT}(; dims_RZ = (length(R1D), length(Z1D)))
    return flf_analysis_field_lines_rz_plane!(flf, R1D, Z1D, BR, BZ, Bϕ, cell_state;
                                       dR, dZ, out_wall_idx)
end

# Convenience dispatch for RAPID object
function flf_analysis_field_lines_rz_plane(RP::RAPID)
    return flf_analysis_field_lines_rz_plane(
        RP.G.R1D, RP.G.Z1D, RP.fields.BR, RP.fields.BZ, RP.fields.Bϕ,
        RP.G.cell_state.>=0; # Use cell_state as boolean mask
        dR=RP.G.dR, dZ=RP.G.dZ,
        out_wall_idx=RP.G.nodes.out_wall_nids
    )
end

function flf_analysis_field_lines_rz_plane!(RP::RAPID)
    return flf_analysis_field_lines_rz_plane!(RP.flf,
        RP.G.R1D, RP.G.Z1D, RP.fields.BR, RP.fields.BZ, RP.fields.Bϕ,
        RP.G.cell_state.>=0; # Use cell_state as boolean mask
        dR=RP.G.dR, dZ=RP.G.dZ,
        out_wall_idx=RP.G.nodes.out_wall_nids
    )
end

# Helper function to create 2D interpolation that matches MATLAB's griddedInterpolant behavior
function my_interpolation(R1D, Z1D, data_2d)
    itp = interpolate((R1D, Z1D), data_2d, Gridded(Linear()))
    # itp = interpolate((R1D, Z1D), data_2d, Gridded(Cubic()))
    # Add extrapolation with constant boundary values
    # return extrapolate(itp, Flat())
    return itp
end
