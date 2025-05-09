"""
Fields module for RAPID2D.

Contains functions related to electromagnetic field calculations, including:
- Vacuum field calculation
- Self-consistent field calculation
- Field line tracing
- Green's function calculations for Grad-Shafranov equation
- External field handling
"""

# Export public functions
export update_fields!,
       update_self_fields!,
       calculate_field_unit_vectors!,
       calculate_parallel_electric_field!,
       flf_analysis_of_field_lines_in_RZ_plane

# Export external field types and functions
export AbstractExternalField, TimeSeriesExternalField
export get_fields_at_time, interpolate_fields

# Required imports for field calculations
using LinearAlgebra

"""
    update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the self-consistent electromagnetic fields based on plasma current.
"""
function update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_self_fields! not fully implemented yet"

    # Just a dummy operation until proper implementation
    RP.fields.BR_self .= zeros(FT, RP.G.NR, RP.G.NZ)
    RP.fields.BZ_self .= zeros(FT, RP.G.NR, RP.G.NZ)

    # Update total fields
    RP.fields.BR .= RP.fields.BR_ext .+ RP.fields.BR_self
    RP.fields.BZ .= RP.fields.BZ_ext .+ RP.fields.BZ_self

    # Recalculate derived fields
    calculate_field_magnitudes!(RP)
    calculate_field_unit_vectors!(RP)

    return RP
end

"""
    calculate_field_magnitudes!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate the magnitudes of the field components.
"""
function calculate_field_magnitudes!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Calculate field magnitudes
    RP.fields.Bpol .= sqrt.(RP.fields.BR.^2 .+ RP.fields.BZ.^2)
    RP.fields.Btot .= sqrt.(RP.fields.Bpol.^2 .+ RP.fields.Bϕ.^2)

    return RP
end

"""
    calculate_field_unit_vectors!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate unit vectors for the magnetic field.
"""
function calculate_field_unit_vectors!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Avoid division by zero
    epsilon = FT(1.0e-10)
    mask = RP.fields.Btot .< epsilon

    b_denominator = copy(RP.fields.Btot)
    b_denominator[mask] .= FT(1.0)

    RP.fields.bR .= RP.fields.BR ./ b_denominator
    RP.fields.bZ .= RP.fields.BZ ./ b_denominator
    RP.fields.bϕ .= RP.fields.Bϕ ./ b_denominator

    # Zero out unit vectors where total field is near zero
    RP.fields.bR[mask] .= FT(0.0)
    RP.fields.bZ[mask] .= FT(0.0)
    RP.fields.bϕ[mask] .= FT(1.0) # Default to toroidal direction

    return RP
end

"""
    calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate parallel electric field components.
"""
function calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Set external parallel E-field
    RP.fields.E_para_ext .= RP.fields.Eϕ_ext .* RP.fields.bϕ

    # Add self-consistent terms if enabled
    if RP.flags.E_para_self_ES
        # In a real implementation, this would be calculated from plasma quantities
    end

    if RP.flags.E_para_self_EM
        # In a real implementation, this would be calculated from time-dependent fields
    end

    # Set total parallel E-field
    RP.fields.E_para_tot .= RP.fields.E_para_ext .+
                             RP.fields.E_para_self_ES .+
                             RP.fields.E_para_self_EM

    return RP
end

"""
    flf_analysis_of_field_lines_in_RZ_plane(RP::RAPID{FT}) where {FT<:AbstractFloat}

Perform field line following (FLF) analysis in the R-Z plane.
Returns a dictionary with field line connection lengths and related data.
"""
function flf_analysis_of_field_lines_in_RZ_plane(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "flf_analysis_of_field_lines_in_RZ_plane not fully implemented yet"

    # Create a placeholder FLF structure
    FLF = Dict{Symbol, Any}(
        :Lpol_forward => zeros(FT, RP.G.NR, RP.G.NZ),
        :Lpol_backward => zeros(FT, RP.G.NR, RP.G.NZ),
        :Lpol_tot => zeros(FT, RP.G.NR, RP.G.NZ),
        :is_closed => zeros(Int, RP.G.NZ*RP.G.NR)
    )

    return FLF
end

"""
    update_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}

Update external fields based on current simulation time or specified time.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `time_s::FT`: Time at which to update fields (default: current simulation time)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance
"""
function update_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}
    # Use manual mode if no external field source is specified
    if RP.external_field === nothing
        if RP.config.device_Name == "manual"
            # Manual settings already applied during initialization, nothing to do
            return RP
        else
            error("No external field source specified")
        end
    end

    # Get external fields at specified time
    extF = get_fields_at_time(RP.external_field, time_s, RP.G)

    # Update field components
    RP.fields.BR_ext .= extF.BR
    RP.fields.BZ_ext .= extF.BZ
    RP.fields.LV_ext .= extF.LV
    RP.fields.psi_ext .= extF.psi

    # Calculate toroidal electric field
    RP.fields.Eϕ_ext .= RP.fields.LV_ext ./ (2π * RP.G.R2D)

    RP.fields.Bϕ = RP.fields.R0B0 ./ (RP.G.R2D);

    # Combine external and self-generated fields
    RP.fields.BR .= RP.fields.BR_ext .+ RP.fields.BR_self
    RP.fields.BZ .= RP.fields.BZ_ext .+ RP.fields.BZ_self
    RP.fields.Eϕ .= RP.fields.Eϕ_ext .+ RP.fields.Eϕ_self
    RP.fields.psi .= RP.fields.psi_ext .+ RP.fields.psi_self

    # Update field-related calculations
    calculate_field_magnitudes!(RP)
    calculate_field_unit_vectors!(RP)
    calculate_parallel_electric_field!(RP)

    return RP
end

# -----------------------------------------------------------------------------
# External Field Types and Functions
# -----------------------------------------------------------------------------

"""
    TimeSeriesExternalField{FT<:AbstractFloat} <: AbstractExternalField{FT}

Stores time series data for external electromagnetic fields.

# Fields
- `time_s::Vector{FT}`: Time points in seconds
- `BR::Array{FT,3}`: Radial magnetic field component [T] (NR×NZ×NT)
- `BZ::Array{FT,3}`: Vertical magnetic field component [T] (NR×NZ×NT)
- `psi::Array{FT,3}`: Magnetic flux [Wb/rad] (NR×NZ×NT)
- `LV::Array{FT,3}`: Loop voltage [V] (NR×NZ×NT)
- `R::Matrix{FT}`: 2D grid of R coordinates [m]
- `Z::Matrix{FT}`: 2D grid of Z coordinates [m]
- `R_NUM::Int`: Number of R grid points
- `Z_NUM::Int`: Number of Z grid points
- `R_MIN::FT`: Minimum R value [m]
- `R_MAX::FT`: Maximum R value [m]
- `Z_MIN::FT`: Minimum Z value [m]
- `Z_MAX::FT`: Maximum Z value [m]
"""
mutable struct TimeSeriesExternalField{FT<:AbstractFloat} <: AbstractExternalField{FT}
    time_s::Vector{FT}    # Time points [s]

    # Field components
    BR::Array{FT,3}       # Radial magnetic field [T]
    BZ::Array{FT,3}       # Vertical magnetic field [T]
    psi::Array{FT,3}      # Magnetic flux [Wb/rad]
    LV::Array{FT,3}       # Loop voltage [V]

    # Grid geometry
    R::Matrix{FT}         # R coordinates [m]
    Z::Matrix{FT}         # Z coordinates [m]
    R_NUM::Int            # Number of R grid points
    Z_NUM::Int            # Number of Z grid points
    R_MIN::FT             # Minimum R value [m]
    R_MAX::FT             # Maximum R value [m]
    Z_MIN::FT             # Minimum Z value [m]
    Z_MAX::FT             # Maximum Z value [m]
end

"""
    get_fields_at_time(field::TimeSeriesExternalField{FT}, time::FT, grid::GridGeometry{FT}) where {FT<:AbstractFloat}

Interpolate field values at a specific time from time series data.

# Arguments
- `field::TimeSeriesExternalField{FT}`: The time series external field data
- `time::FT`: The time at which to get field values
- `grid::GridGeometry{FT}`: Grid geometry (not used for TimeSeriesExternalField but included for API consistency)

# Returns
- `NamedTuple`: Contains fields BR, BZ, LV, psi interpolated at the specified time
"""
function get_fields_at_time(field::TimeSeriesExternalField{FT}, time::FT, grid=nothing) where {FT<:AbstractFloat}
    # Find the time indices for interpolation
    if time <= field.time_s[1]
        # Before first time point - use first time point
        idx = 1
        t_weight = FT(0)
    elseif time >= field.time_s[end]
        # After last time point - use last time point
        idx = length(field.time_s) - 1
        t_weight = FT(1)
    else
        # Find the appropriate time interval
        idx = searchsortedlast(field.time_s, time)
        # Calculate interpolation weight
        t_weight = (time - field.time_s[idx]) / (field.time_s[idx+1] - field.time_s[idx])
    end

    # Linear interpolation in time
    BR = (1 - t_weight) * field.BR[:, :, idx] + t_weight * field.BR[:, :, idx+1]
    BZ = (1 - t_weight) * field.BZ[:, :, idx] + t_weight * field.BZ[:, :, idx+1]
    psi = (1 - t_weight) * field.psi[:, :, idx] + t_weight * field.psi[:, :, idx+1]
    LV = (1 - t_weight) * field.LV[:, :, idx] + t_weight * field.LV[:, :, idx+1]

    return (
        BR = BR,
        BZ = BZ,
        LV = LV,
        psi = psi,
        time_s = time
    )
end

"""
    interpolate_fields(ef::TimeSeriesExternalField{FT}, time_s::FT) where {FT<:AbstractFloat}

Linearly interpolates field values from stored data at the given time.

# Returns
- `NamedTuple`: Contains fields BR, BZ, LV, psi interpolated at the specified time
"""
function interpolate_fields(ef::TimeSeriesExternalField{FT}, time_s::FT) where {FT<:AbstractFloat}
    # Check time range
    if time_s < ef.time_s[1] || time_s > ef.time_s[end]
        @warn "Time $(time_s)s is outside the data range [$(ef.time_s[1]), $(ef.time_s[end])]s"
    end

    # Find time index for interpolation
    idx = searchsortedlast(ef.time_s, time_s)
    if idx == 0
        idx = 1
    elseif idx == length(ef.time_s)
        idx = length(ef.time_s) - 1
    end

    # Calculate linear interpolation weight
    t1, t2 = ef.time_s[idx], ef.time_s[idx+1]
    w = (time_s - t1) / (t2 - t1)

    # Interpolate field values
    BR = @. (1-w) * ef.BR[:,:,idx] + w * ef.BR[:,:,idx+1]
    BZ = @. (1-w) * ef.BZ[:,:,idx] + w * ef.BZ[:,:,idx+1]
    LV = @. (1-w) * ef.LV[:,:,idx] + w * ef.LV[:,:,idx+1]
    psi = @. (1-w) * ef.psi[:,:,idx] + w * ef.psi[:,:,idx+1]

    return (
        BR = BR,
        BZ = BZ,
        LV = LV,
        psi = psi,
        time_s = time_s
    )
end
