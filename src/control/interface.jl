"""
Interface functions between RAPID2D simulation and controllers.

Provides functions to extract measurements from RAPID2D and perform control updates.
"""

"""
    extract_plasma_current(RP::RAPID{FT}) where {FT<:AbstractFloat}

Extract current plasma current from RAPID simulation.
This is the interface function between RAPID2D and the current controller.

Returns the total toroidal current in Amperes.
"""
function extract_plasma_current(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Calculate toroidal current: Itor = ∫∫ Jϕ * dR * dZ
    Itor = FT(0.0)

    for i in 1:RP.G.nR
        for j in 1:RP.G.nZ
            if RP.G.inVol2D[i, j]  # Only inside plasma volume
                Itor += RP.plasma.Jϕ[i, j] * RP.G.dR * RP.G.dZ
            end
        end
    end

    return Itor
end

"""
    extract_plasma_position(RP::RAPID{FT}) where {FT<:AbstractFloat}

Extract current plasma position (current-weighted average R position).

Returns the plasma center R position in meters.
"""
function extract_plasma_position(RP::RAPID{FT}) where {FT<:AbstractFloat}
    total_Jϕ = sum(RP.plasma.Jϕ)
    if total_Jϕ != 0
        plasma_center_R = sum(RP.plasma.Jϕ .* RP.G.R2D) / total_Jϕ
    else
        plasma_center_R = mean(RP.G.R1D)  # fallback to geometric center
    end
    return plasma_center_R
end

"""
    extract_plasma_temperature(RP::RAPID{FT}) where {FT<:AbstractFloat}

Extract average plasma temperature.

Returns the volume-averaged plasma temperature in eV.
"""
function extract_plasma_temperature(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Volume-weighted average temperature
    total_volume = sum(RP.G.inVol2D)
    if total_volume > 0
        avg_temp = sum(RP.plasma.Te .* RP.G.inVol2D) / total_volume
    else
        avg_temp = FT(0.0)
    end
    return avg_temp
end

"""
    update_controller!(RP::RAPID{FT}, controller::Controller{FT}) where {FT<:AbstractFloat}

Main control loop function: measure, compute control, apply signal.

This is the main interface function called from the simulation loop.

# Arguments
- `RP`: RAPID simulation object
- `controller`: Controller instance
"""
function update_controller!(RP::RAPID{FT}, controller::Controller{FT}) where {FT<:AbstractFloat}
    # Extract measurement based on control type
    current_value = if controller.control_type == "current"
        extract_plasma_current(RP)
    elseif controller.control_type == "position"
        extract_plasma_position(RP)
    elseif controller.control_type == "temperature"
        extract_plasma_temperature(RP)
    else
        @warn "Unknown control type: $(controller.control_type). Using plasma current as fallback."
        extract_plasma_current(RP)
    end

    # Compute control signal using PID
    control_signal = compute_control_signal!(controller, current_value)

    # Apply control signal to coils
    apply_control_signal!(controller, control_signal)

    return RP
end

# Export interface functions
export extract_plasma_current, extract_plasma_position, extract_plasma_temperature,
       update_controller!
