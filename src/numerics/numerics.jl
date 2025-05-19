"""
    smooth_data_2D!(target::AbstractMatrix{FT};
                    num_SM::Int=1, weighting::Union{AbstractMatrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}

In-place smoothing of a 2D field using a weighted average filter. Modifies the input array directly.

# Arguments
- `target::AbstractMatrix{FT}`: The field to be smoothed (modified in-place)
- `num_SM::Int=1`: Number of smoothing iterations (if 0, no smoothing is applied)
- `weighting::Union{AbstractMatrix{FT},Nothing}=nothing`: Optional weighting field (defaults to ones if not specified)

# Returns
- `target::AbstractMatrix{FT}`: The smoothed field (same as input `target`)

# Notes
- Uses a weighted average stencil that respects field geometry through the weighting field
- Multiple passes can be applied by setting `num_SM` > 1
- Areas with zero weight maintain their original values
- Setting `num_SM=0` returns the original field unchanged
- Modifies the input array directly to minimize allocations
"""
function smooth_data_2D!(target::AbstractMatrix{FT};
                         num_SM::Int=1, weighting::AbstractMatrix{FT}=ones(FT, size(target))) where {FT<:AbstractFloat}
    # If no smoothing requested, return original field unchanged
    if num_SM <= 0
        return target
    end

    NR, NZ = size(target)  # Note: we use first index for R, second for Z

    # Create arrays with ghost cells (to handle boundaries)
    VV = zeros(FT, NR+2, NZ+2)
    VV[2:end-1, 2:end-1] .= weighting

    # Initialize the target array with ghost cells
    tarVal = zeros(FT, NR+2, NZ+2)
    tarVal[2:end-1, 2:end-1] .= target

    # Create temporary array for smoothing iterations
    smoothed = similar(tarVal)

    # Define interior indices
    rid = 2:NR+1
    zid = 2:NZ+1

    # Calculate smoothing coefficients
    Z_coeff_down = @. 0.5 * VV[rid, zid-1] / (VV[rid, zid-1] + VV[rid, zid])
    Z_coeff_middle = @. 0.5 * VV[rid, zid] * (VV[rid, zid-1] + 2.0*VV[rid, zid] + VV[rid, zid+1]) /
                        ((VV[rid, zid-1] + VV[rid, zid]) * (VV[rid, zid] + VV[rid, zid+1]))
    Z_coeff_up = @. 0.5 * VV[rid, zid+1] / (VV[rid, zid+1] + VV[rid, zid])

    R_coeff_left = @. 0.5 * VV[rid-1, zid] / (VV[rid-1, zid] + VV[rid, zid])
    R_coeff_middle = @. 0.5 * VV[rid, zid] * (VV[rid-1, zid] + 2.0*VV[rid, zid] + VV[rid+1, zid]) /
                        ((VV[rid-1, zid] + VV[rid, zid]) * (VV[rid, zid] + VV[rid+1, zid]))
    R_coeff_right = @. 0.5 * VV[rid+1, zid] / (VV[rid+1, zid] + VV[rid, zid])

    # Handle zero weighting areas specially
    idx_zero_weighting = findall(weighting .== 0)
    if !isempty(idx_zero_weighting)
        Z_coeff_down[idx_zero_weighting] .= 0
        Z_coeff_middle[idx_zero_weighting] .= 1
        Z_coeff_up[idx_zero_weighting] .= 0

        R_coeff_left[idx_zero_weighting] .= 0
        R_coeff_middle[idx_zero_weighting] .= 1
        R_coeff_right[idx_zero_weighting] .= 0
    end

    # Apply smoothing iterations
    for i in 1:num_SM
        # Z-direction smoothing
        @inbounds for j in zid
            for i in rid
                smoothed[i, j] = Z_coeff_down[i-1, j-1] * tarVal[i, j-1] +
                                Z_coeff_middle[i-1, j-1] * tarVal[i, j] +
                                Z_coeff_up[i-1, j-1] * tarVal[i, j+1]
            end
        end

        # Copy results back to tarVal for next iteration
        tarVal[rid, zid] .= smoothed[rid, zid]

        # R-direction smoothing
        @inbounds for j in zid
            for i in rid
                smoothed[i, j] = R_coeff_left[i-1, j-1] * tarVal[i-1, j] +
                               R_coeff_middle[i-1, j-1] * tarVal[i, j] +
                               R_coeff_right[i-1, j-1] * tarVal[i+1, j]
            end
        end

        # Copy results back to tarVal for next iteration
        tarVal[rid, zid] .= smoothed[rid, zid]
    end

    # Copy the smoothed interior back to target (in-place modification)
    target .= tarVal[rid, zid]

    return target
end

"""
    smooth_data_2D(target::AbstractMatrix{FT};
                   num_SM::Int=1, weighting::Union{AbstractMatrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}

Smooth a 2D field using a weighted average filter.

# Arguments
- `target::AbstractMatrix{FT}`: The field to be smoothed
- `num_SM::Int=1`: Number of smoothing iterations (if 0, returns the original field)
- `weighting::Union{AbstractMatrix{FT},Nothing}=nothing`: Optional weighting field (defaults to ones if not specified)

# Returns
- `AbstractMatrix{FT}`: The smoothed field (new allocation)

# Notes
- Uses a weighted average stencil that respects field geometry through the weighting field
- Multiple passes can be applied by setting `num_SM` > 1
- Areas with zero weight maintain their original values
- Setting `num_SM=0` bypasses smoothing and returns a copy of the original field
"""
function smooth_data_2D(target::AbstractMatrix{FT};
                        num_SM::Int=1, weighting::AbstractMatrix{FT}=ones(FT, size(target))) where {FT<:AbstractFloat}
    # If no smoothing requested, return a copy of the original field
    if num_SM <= 0
        return copy(target)
    end

    # Create a copy and then smooth it in-place
    result = copy(target)
    return smooth_data_2D!(result; num_SM, weighting)
end

"""
    smooth_data_2D_efficient!(target::Matrix{FT};
                              num_SM::Int=1, weighting::Union{Matrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}

Highly optimized in-place smoothing of a 2D field using a weighted average filter.
Minimizes allocations by using explicit loops and computing coefficients on-the-fly.

# Arguments
- `target::Matrix{FT}`: The field to be smoothed (modified in-place)
- `num_SM::Int=1`: Number of smoothing iterations (if 0, no smoothing is applied)
- `weighting::Union{Matrix{FT},Nothing}=nothing`: Optional weighting field (defaults to ones if not specified)

# Returns
- `target::Matrix{FT}`: The smoothed field (same as input `target`)

# Notes
- Uses a weighted average stencil that respects field geometry through the weighting field
- Multiple passes can be applied by setting `num_SM` > 1
- Areas with zero weight maintain their original values
- Setting `num_SM=0` returns the original field unchanged
- Minimizes allocations by avoiding temporary arrays and using explicit loops
"""
function smooth_data_2D_efficient!(target::Matrix{FT};
                                  num_SM::Int=1, weighting::Union{Matrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}
    # If no smoothing requested, return original field unchanged
    if num_SM <= 0
        return target
    end

    NR, NZ = size(target)

    # Short circuit for trivial sizes
    if NR <= 2 || NZ <= 2
        return target
    end

    # Choose weighting - use ones by default if not specified
    weight = isnothing(weighting) ? ones(FT, size(target)) : weighting
    size(weight) == size(target) || throw(DimensionMismatch("weighting array must have same dimensions as target"))

    # Create a single row buffer for intermediate calculations to minimize allocations
    buffer = Vector{FT}(undef, max(NR, NZ))

    # Keep a copy of the original for each smoothing pass
    # This is necessary to avoid propagating changes within a single smoothing iteration
    orig = copy(target)

    # Constants for coefficient calculations
    half = FT(0.5)
    two = FT(2.0)
    zero_FT = zero(FT)
    one_FT = one(FT)

    # Apply smoothing iterations
    for _ in 1:num_SM
        # Copy original data to work with
        copyto!(target, orig)

        # Z-direction smoothing
        for i in 2:NR-1
            # First, gather all inputs needed for this row into the buffer to minimize memory access
            @inbounds for j in 1:NZ
                buffer[j] = target[i, j]
            end

            # Now process the interior points with pre-fetched data
            @inbounds for j in 2:NZ-1
                # Skip if weight is zero (maintain original value)
                w_center = weight[i, j]
                if w_center == zero_FT
                    continue
                end

                # Calculate coefficients on-the-fly
                w_down = weight[i, j-1]
                w_up = weight[i, j+1]

                # Compute coefficient for point below
                denom_down = w_down + w_center
                c_down = denom_down > zero_FT ? half * w_down / denom_down : zero_FT

                # Compute coefficient for current point
                denom1 = denom_down > zero_FT ? denom_down : one_FT
                denom2 = (w_center + w_up) > zero_FT ? (w_center + w_up) : one_FT
                c_middle = half * w_center * (w_down + two*w_center + w_up) / (denom1 * denom2)

                # Compute coefficient for point above
                denom_up = w_up + w_center
                c_up = denom_up > zero_FT ? half * w_up / denom_up : zero_FT

                # Apply smoothing using buffered values
                orig[i, j] = c_down * buffer[j-1] + c_middle * buffer[j] + c_up * buffer[j+1]
            end
        end

        # We've updated orig with Z-direction smoothing; now switch to R-direction
        copyto!(target, orig)

        # R-direction smoothing
        for j in 2:NZ-1
            # First, gather all inputs needed for this column into the buffer
            @inbounds for i in 1:NR
                buffer[i] = target[i, j]
            end

            # Now process the interior points with pre-fetched data
            @inbounds for i in 2:NR-1
                # Skip if weight is zero (maintain original value)
                w_center = weight[i, j]
                if w_center == zero_FT
                    continue
                end

                # Calculate coefficients on-the-fly
                w_left = weight[i-1, j]
                w_right = weight[i+1, j]

                # Compute coefficient for point to the left
                denom_left = w_left + w_center
                c_left = denom_left > zero_FT ? half * w_left / denom_left : zero_FT

                # Compute coefficient for current point
                denom1 = denom_left > zero_FT ? denom_left : one_FT
                denom2 = (w_center + w_right) > zero_FT ? (w_center + w_right) : one_FT
                c_middle = half * w_center * (w_left + two*w_center + w_right) / (denom1 * denom2)

                # Compute coefficient for point to the right
                denom_right = w_right + w_center
                c_right = denom_right > zero_FT ? half * w_right / denom_right : zero_FT

                # Apply smoothing using buffered values
                orig[i, j] = c_left * buffer[i-1] + c_middle * buffer[i] + c_right * buffer[i+1]
            end
        end

        # After each full smoothing iteration, copy results back to target
        copyto!(target, orig)
    end

    return target
end

# Also add an allocating version of the efficient implementation
"""
    smooth_data_2D_efficient(target::Matrix{FT};
                             num_SM::Int=1, weighting::Union{Matrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}

Allocation-efficient smoothing of a 2D field using a weighted average filter, returning a new array.

# Arguments
- `target::Matrix{FT}`: The field to be smoothed
- `num_SM::Int=1`: Number of smoothing iterations (if 0, returns a copy of the original field)
- `weighting::Union{Matrix{FT},Nothing}=nothing`: Optional weighting field (defaults to ones if not specified)

# Returns
- `Matrix{FT}`: The smoothed field (new allocation)

# Notes
- Uses a weighted average stencil that respects field geometry through the weighting field
- Uses optimized algorithms to minimize temporary allocations
- Multiple passes can be applied by setting `num_SM` > 1
- Areas with zero weight maintain their original values
"""
function smooth_data_2D_efficient(target::Matrix{FT};
                                 num_SM::Int=1, weighting::Union{Matrix{FT},Nothing}=nothing) where {FT<:AbstractFloat}
    # If no smoothing requested, return a copy of the original field
    if num_SM <= 0
        return copy(target)
    end

    # Create a copy and then smooth it in-place with our efficient implementation
    result = copy(target)
    return smooth_data_2D_efficient!(result; num_SM=num_SM, weighting=weighting)
end

# Export the additional efficient function
export smooth_data_2D_efficient