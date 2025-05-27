
"""
    update_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update 0D diagnostic snapshots in the RAPID object.
Note: This function updates what were previously called "1D" diagnostics,
which are actually 0D (scalar) quantities averaged over the volume.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update
"""
function update_snap0D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Get current index
    idx = RP.diagnostics.snap0D.idx

    # Check if we've reached the end of the array
    if idx > length(RP.diagnostics.snap0D.time_s)
        @warn "Reached end of pre-allocated 0D diagnostic arrays at index $idx"
        return nothing
    end

    # Store current time
    RP.diagnostics.snap0D.time_s[idx] = RP.time_s

    # Calculate and store 0D diagnostics
    # Average electron density
    RP.diagnostics.snap0D.ne_avg[idx] = sum(RP.plasma.ne .* RP.G.inVol2D) / RP.G.device_inVolume

    # Average electron energy
    avg_eErg_eV = sum(1.5 * RP.plasma.Te_eV .* RP.plasma.ne .* RP.G.inVol2D) /
                 sum(RP.plasma.ne .* RP.G.inVol2D)
    RP.diagnostics.snap0D.avg_mean_eErg_eV[idx] = avg_eErg_eV

    # Average parallel electric fields
    RP.diagnostics.snap0D.avg_Epara_ext[idx] = sum(RP.fields.E_para_ext .* RP.G.inVol2D) / RP.G.device_inVolume
    RP.diagnostics.snap0D.avg_Epara_tot[idx] = sum(RP.fields.E_para_tot .* RP.G.inVol2D) / RP.G.device_inVolume

    # Calculate total toroidal current (simplified)
    RP.diagnostics.snap0D.I_tor[idx] = sum(RP.plasma.Jϕ .* RP.G.inVol2D)

    # Increment index
    RP.diagnostics.snap0D.idx += 1

    return nothing
end

"""
    update_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update 2D diagnostic snapshots in the RAPID object.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance to update
"""
function update_snap2D!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Get current index
    idx = RP.diagnostics.snap2D.idx

    # Check if we've reached the end of the array
    if idx > length(RP.diagnostics.snap2D.time_s)
        @warn "Reached end of pre-allocated 2D diagnostic arrays at index $idx"
        return nothing
    end

    # Store current time
    RP.diagnostics.snap2D.time_s[idx] = RP.time_s

    # Store 2D field snapshots using pre-allocated 3D arrays
    # The arrays are already allocated with proper dimensions (NR, NZ, dim_tt_2D)
    RP.diagnostics.snap2D.ne[:, :, idx] = RP.plasma.ne
    RP.diagnostics.snap2D.Te_eV[:, :, idx] = RP.plasma.Te_eV
    RP.diagnostics.snap2D.ue_para[:, :, idx] = RP.plasma.ue_para
    RP.diagnostics.snap2D.E_para_tot[:, :, idx] = RP.fields.E_para_tot

    # Increment index
    RP.diagnostics.snap2D.idx += 1

    return nothing
end
