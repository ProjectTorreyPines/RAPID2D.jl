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
export update_external_fields!,
       update_self_fields!,
       calculate_field_unit_vectors!,
       calculate_parallel_electric_field!,
       flf_analysis_of_field_lines_in_RZ_plane

# Export external field types and functions
export AbstractExternalField, ExternalFieldData
export get_fields_at_time, interpolate_fields, read_external_field_data

"""
    update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the self-consistent electromagnetic fields based on plasma current.
"""
function update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_self_fields! not fully implemented yet"

    # Just a dummy operation until proper implementation
    RP.fields.BR_self .= zeros(FT, RP.G.NZ, RP.G.NR)
    RP.fields.BZ_self .= zeros(FT, RP.G.NZ, RP.G.NR)

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
        :Lpol_forward => zeros(FT, RP.G.NZ, RP.G.NR),
        :Lpol_backward => zeros(FT, RP.G.NZ, RP.G.NR),
        :Lpol_tot => zeros(FT, RP.G.NZ, RP.G.NR),
        :is_closed => zeros(Int, RP.G.NZ*RP.G.NR)
    )

    return FLF
end

"""
    update_external_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update external fields based on current simulation time.
"""
function update_external_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Use manual mode if no external field source is specified
    if RP.external_field === nothing
        if RP.config.device_Name == "manual"
            # Manual settings already applied during initialization, nothing to do
            return
        else
            error("No external field source specified")
        end
    end

    # Get external fields at current time
    BR_ext, BZ_ext, LV_ext, psi_ext = get_fields_at_time(
        RP.external_field, RP.time_s, RP.G)

    # Update field components
    RP.fields.BR_ext .= BR_ext
    RP.fields.BZ_ext .= BZ_ext
    RP.fields.LV_ext .= LV_ext
    RP.fields.psi_ext .= psi_ext

    # Calculate toroidal electric field
    RP.fields.Eϕ_ext .= RP.fields.LV_ext ./ (2π * RP.G.R2D)

    # Combine external and self-generated fields
    RP.fields.BR .= RP.fields.BR_ext .+ RP.fields.BR_self
    RP.fields.BZ .= RP.fields.BZ_ext .+ RP.fields.BZ_self
    RP.fields.Eϕ .= RP.fields.Eϕ_ext .+ RP.fields.Eϕ_self
    RP.fields.psi .= RP.fields.psi_ext .+ RP.fields.psi_self

    # Calculate parallel electric field
    RP.fields.E_para_ext .= RP.fields.Eϕ_ext .* RP.fields.bϕ

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
    ExternalFieldData{FT<:AbstractFloat} <: AbstractExternalField{FT}

Stores pre-computed external electromagnetic field data from files or other sources.
Provides interpolated BR, BZ, LV, psi values at specific times.
"""
mutable struct ExternalFieldData{FT<:AbstractFloat} <: AbstractExternalField{FT}
    # Time data
    time_s::Vector{FT}

    # Field data - 3D arrays (Z, R, time)
    BR::Array{FT, 3}         # Radial magnetic field [T]
    BZ::Array{FT, 3}         # Vertical magnetic field [T]
    LV::Array{FT, 3}         # Loop Voltage [V]
    psi::Array{FT, 3}        # Magnetic flux [Wb/rad]

    # Metadata
    description::String      # Data description

    # Internal constructor
    function ExternalFieldData{FT}(
        time_s::Vector{FT},
        BR::Array{FT, 3},
        BZ::Array{FT, 3},
        LV::Array{FT, 3},
        psi::Array{FT, 3};
        description::String = ""
    ) where FT<:AbstractFloat
        # Dimension validation
        if size(BR, 3) != length(time_s) ||
           size(BZ, 3) != length(time_s) ||
           size(LV, 3) != length(time_s) ||
           size(psi, 3) != length(time_s)
            error("Time dimension must match the length of time_s vector")
        end

        return new{FT}(time_s, BR, BZ, LV, psi, description)
    end
end

# External constructor (allows type inference)
function ExternalFieldData(
    time_s::Vector{FT},
    BR::Array{FT, 3},
    BZ::Array{FT, 3},
    LV::Array{FT, 3},
    psi::Array{FT, 3};
    description::String = ""
) where FT<:AbstractFloat
    return ExternalFieldData{FT}(time_s, BR, BZ, LV, psi; description=description)
end

"""
    interpolate_fields(ef::ExternalFieldData{FT}, time_s::FT) where {FT<:AbstractFloat}

Linearly interpolates field values from stored data at the given time.
"""
function interpolate_fields(ef::ExternalFieldData{FT}, time_s::FT) where {FT<:AbstractFloat}
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

    return BR, BZ, LV, psi
end

"""
    get_fields_at_time(ef::AbstractExternalField{FT}, time_s::FT, grid::GridGeometry{FT})

Returns external field components at the specified time.
All AbstractExternalField types must implement this function.
"""
function get_fields_at_time(ef::ExternalFieldData{FT}, time_s::FT, grid::GridGeometry{FT}) where {FT<:AbstractFloat}
    return interpolate_fields(ef, time_s)
end

"""
    read_external_field_data(file_path::String, ::Type{FT}=Float64) where {FT<:AbstractFloat}

Reads external electromagnetic field data from a file and creates an ExternalFieldData object.
"""
function read_external_field_data(file_path::String, ::Type{FT}=Float64) where {FT<:AbstractFloat}
    # Currently a simple implementation, needs proper implementation based on actual file format
    error("The read_external_field_data function is not yet implemented")
end
