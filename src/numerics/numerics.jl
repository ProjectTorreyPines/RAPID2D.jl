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
    @timeit RAPID_TIMER "smooth_data_2D!" begin
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