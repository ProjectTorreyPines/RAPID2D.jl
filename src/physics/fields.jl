"""
Fields module for RAPID2D.

Contains functions related to electromagnetic field calculations, including:
- Vacuum field calculation
- Self-consistent field calculation
- Field line tracing
- Green's function calculations for Grad-Shafranov equation
"""

# Export public functions
export update_vacuum_fields!,
       update_self_fields!,
       calculate_field_unit_vectors!,
       calculate_parallel_electric_field!,
       flf_analysis_of_field_lines_in_RZ_plane

"""
    update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the self-consistent electromagnetic fields based on plasma current.
"""
function update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_self_fields! not fully implemented yet"

    # Just a dummy operation until proper implementation
    RP.fields.BR_self .= zeros(FT, RP.NZ, RP.NR)
    RP.fields.BZ_self .= zeros(FT, RP.NZ, RP.NR)

    # Update total fields
    RP.fields.BR .= RP.fields.BR_vac .+ RP.fields.BR_self
    RP.fields.BZ .= RP.fields.BZ_vac .+ RP.fields.BZ_self

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
    # Set vacuum parallel E-field
    RP.fields.E_para_vac .= RP.fields.Eϕ .* RP.fields.bϕ

    # Add self-consistent terms if enabled
    if RP.flags.E_para_self_ES
        # In a real implementation, this would be calculated from plasma quantities
    end

    if RP.flags.E_para_self_EM
        # In a real implementation, this would be calculated from time-dependent fields
    end

    # Set total parallel E-field
    RP.fields.E_para_tot .= RP.fields.E_para_vac .+
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
        :Lpol_forward => zeros(FT, RP.NZ, RP.NR),
        :Lpol_backward => zeros(FT, RP.NZ, RP.NR),
        :Lpol_tot => zeros(FT, RP.NZ, RP.NR),
        :is_closed => zeros(Int, RP.NZ*RP.NR)
    )

    return FLF
end