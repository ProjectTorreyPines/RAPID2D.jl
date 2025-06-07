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
       combine_external_and_self_fields!,
       calculate_magnetic_field_unit_vectors!

# Export external field types and functions
export AbstractExternalField, TimeSeriesExternalField
export calculate_external_fields_at_time

# Required imports for field calculations
using LinearAlgebra

"""
    update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Update the self-consistent electromagnetic fields based on plasma current.
"""
function update_self_fields!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Placeholder implementation - will be filled in later
    @warn "update_self_fields! not fully implemented yet"

    F = RP.fields

    # Just a dummy operation until proper implementation
    F.BR_self .= zeros(FT, RP.G.NR, RP.G.NZ)
    F.BZ_self .= zeros(FT, RP.G.NR, RP.G.NZ)

    F.Eϕ_self .= zeros(FT, RP.G.NR, RP.G.NZ)

    return RP
end


"""
    calculate_derived_magnetic_field_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate derived magnetic field quantities based on the current field values.
"""
function calculate_derived_magnetic_field_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
    @. RP.fields.Btot = sqrt(RP.fields.Bpol^2 + RP.fields.Bϕ^2)

    calculate_magnetic_field_unit_vectors!(RP)

    return RP
end

"""
    calculate_magnetic_field_unit_vectors!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate unit vectors for the magnetic field.
"""
function calculate_magnetic_field_unit_vectors!(RP::RAPID{FT}) where {FT<:AbstractFloat}

    F = RP.fields

    # Unit vectors for magnetic field
    @. F.bR = F.BR / F.Btot
    @. F.bZ = F.BZ / F.Btot
    @. F.bϕ = F.Bϕ / F.Btot

    # Poloidal magnetic field unit vectors
    @. F.bpol_R = F.BR / F.Bpol
    @. F.bpol_Z = F.BZ / F.Bpol

    # Avoid division by zero for poloidal unit vectors
    mask = (F.Bpol .== 0)
    F.bpol_R[mask] .= FT(0.0)
    F.bpol_Z[mask] .= FT(0.0)

    return RP
end

"""
    calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate the parallel electric field based on external and self-generated fields.
"""
function calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    F = RP.fields
    # external parallel electric field
    @. F.E_para_ext = F.Eϕ_ext * F.bϕ

    # self parallel electric field
    if RP.flags.E_para_self_ES
        @. F.E_para_self_ES = sign(-F.Eϕ_ext*F.Bϕ) * F.Epol_self * F.Bpol / F.Btot
    end
    if RP.flags.E_para_self_EM
        @. F.E_para_self_EM = F.Eϕ_self * F.bϕ
    end

    # combine parallel electric fields
    @. F.E_para_tot = F.E_para_ext + F.E_para_self_ES + F.E_para_self_EM

    return RP
end

"""
    update_external_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}

Update external electromagnetic fields based on the current simulation time or a specified time.
This function handles interpolation from time series data and calculates derived external fields.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `time_s::FT`: Time at which to update fields (default: current simulation time)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance
"""
function update_external_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}
    F = RP.fields

    # Use manual mode if no external field source is specified
    if !isnothing(RP.external_field)
        # Get external fields at specified time
        extF = calculate_external_fields_at_time(RP.external_field, time_s)

        # Update field components
        F.BR_ext .= extF.BR
        F.BZ_ext .= extF.BZ
        F.LV_ext .= extF.LV
        F.ψ_ext .= extF.psi
    end

    # Set toroidal magnetic field
    F.Bϕ = F.R0B0 ./ RP.G.R2D

    # Calculate toroidal electric field
    F.Eϕ_ext .= F.LV_ext ./ (2π * RP.G.R2D)

    if RP.flags.Damp_Transp_outWall
        @. F.LV_ext *= RP.damping_func
        @. F.Eϕ_ext *= RP.damping_func
    end

    return RP
end

"""
    combine_external_and_self_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}

Combine external and self-generated electromagnetic fields and calculate derived field quantities.
This function does NOT update the fields themselves, but assumes they are already updated separately.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `time_s::FT`: Time parameter (used only for documentation, not actually used in calculations)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance with combined fields and derived quantities

# Note
External fields should be updated separately using `update_external_fields!` before calling this function.
Self-generated fields should be updated using appropriate physics functions.
"""
function combine_external_and_self_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}
    @timeit RAPID_TIMER "combine_external_and_self_fields!" begin
        F = RP.fields

        # Combine external and self-generated fields
        @. F.BR = F.BR_ext + F.BR_self
        @. F.BZ = F.BZ_ext + F.BZ_self
        @. F.Eϕ = F.Eϕ_ext + F.Eϕ_self
        @. F.ψ = F.ψ_ext + F.ψ_self

        # Update derived magnetic field quantities (Bpol, Btot, unit b vector)
        calculate_derived_magnetic_field_quantities!(RP)
        calculate_parallel_electric_field!(RP)
        return RP
    end
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
    calculate_external_fields_at_time(field::TimeSeriesExternalField{FT}, time::FT, grid::GridGeometry{FT}) where {FT<:AbstractFloat}

Interpolate field values at a specific time from time series data.

# Arguments
- `field::TimeSeriesExternalField{FT}`: The time series external field data
- `time::FT`: The time at which to get field values

# Returns
- `NamedTuple`: Contains fields BR, BZ, LV, psi interpolated at the specified time
"""
function calculate_external_fields_at_time(extF::TimeSeriesExternalField{FT}, time::FT) where {FT<:AbstractFloat}
    # Find the time indices for interpolation
    if time <= extF.time_s[1]
        # Before first time point - use first time point
        idx = 1
        t_weight = FT(0)
    elseif time >= extF.time_s[end]
        # After last time point - use last time point
        idx = length(extF.time_s) - 1
        t_weight = FT(1)
    else
        # Find the appropriate time interval
        idx = searchsortedlast(extF.time_s, time)
        # Calculate interpolation weight
        t_weight = (time - extF.time_s[idx]) / (extF.time_s[idx+1] - extF.time_s[idx])
    end

    # Linear interpolation in time
    BR = (1 - t_weight) * extF.BR[:, :, idx] + t_weight * extF.BR[:, :, idx+1]
    BZ = (1 - t_weight) * extF.BZ[:, :, idx] + t_weight * extF.BZ[:, :, idx+1]
    psi = (1 - t_weight) * extF.psi[:, :, idx] + t_weight * extF.psi[:, :, idx+1]
    LV = (1 - t_weight) * extF.LV[:, :, idx] + t_weight * extF.LV[:, :, idx+1]

    return (
        BR = BR,
        BZ = BZ,
        LV = LV,
        psi = psi,
        time_s = time
    )
end

# -----------------------------------------------------------------------------
# Electrostatic Field Effects Functions
# -----------------------------------------------------------------------------

"""
    my_sigmf(x, a, c)

Sigmoid function equivalent to MATLAB's my_sigmf.
Returns y = 1./(1+exp(-a*(x-c)))

# Arguments
- `x`: Input values
- `a`: Steepness parameter
- `c`: Center point
"""
function my_sigmf(x, a, c)
    return @. 1.0 / (1.0 + exp(-a * (x - c)))
end

"""
    estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Estimate self-consistent electrostatic field effects on plasma transport.

This function ports the MATLAB `Estimate_E_self_effects` function from lines 544-679
in c_RAPID.m. It calculates:
- Shape factor (γ_shape_fac) based on magnetic field properties
- Critical densities for parallel and perpendicular transport
- Self-consistent electric fields (E_self_pol, E_para_self_ES)
- Mean ExB drift velocities
- Turbulent diffusion coefficients

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation object

# Returns
- `RP::RAPID{FT}`: Updated RAPID object with calculated field effects
"""
function estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT<:AbstractFloat}
    # Extract commonly used variables
    pla = RP.plasma
    F = RP.fields
    G = RP.G

    # Physical constants
    cnst = RP.config.constants
    @unpack me, mi, ee, eps0 = RP.config.constants

    # Initialize arrays for calculations
    NR, NZ = G.NR, G.NZ

    # =========================================================================
    # 1. Calculate shape factor (γ_shape_fac) based on magnetic field properties
    # =========================================================================

    Bpol_norm = F.Bpol ./ mean(F.Bpol[G.nodes.in_wall_nids])
    # tan_θB = F.Bpol ./ F.Bϕ
    # tan_θB_norm = tan_θB ./ maximum(tan_θB[G.nodes.in_wall_nids])

    # # Calculate shape factor using sigmoid functions (MATLAB lines 554-556)
    # term1 = my_sigmf(FT(1e4) * Bpol_norm, FT(20.0), FT(0.4))
    # term2 = my_sigmf(tan_θB_norm, FT(5.0), FT(0.4))
    # pla.γ_shape_fac = @. term1 * term2

    pla.γ_shape_fac = min.(FT(1.0), 0.65*Bpol_norm)

    # TODO: implement the following line, when closed surface is formed
    # obj.gamma_coeff(obj.idx_closed_surface) = 0.0; % closed surface has no self-E effects

    # =========================================================================
    # 2. Calculate critical densities
    # =========================================================================
    @. pla.nc_para = (eps0 / (ee * pla.Te_eV)) * (F.Bϕ / F.Bpol)^FT(2.0) * (F.Eϕ / pla.γ_shape_fac)^FT(2.0)
    @. pla.nc_perp = eps0 / me * (F.Btot * F.Bpol / F.Bϕ)^FT(2.0) * (FT(1.0) / pla.γ_shape_fac)^FT(2.0)

    # =========================================================================
    # 3. Calculate self-consistent electric fields (MATLAB lines 582-620)
    # =========================================================================
    # Parallel electrostatic self-field component
    # E_para_self_ES = E_self_pol * (Bpol/Btot) * sign(direction)
    E_pol_required_for_cancellation = @. abs(F.E_para_ext + RP.flags.E_para_self_EM * F.E_para_self_EM)*(F.Btot/F.Bpol);

    ne_SM = smooth_data_2D(pla.ne; num_SM=2, weighting=RP.G.inVol2D)
    # E_self_debye = @. sqrt(abs(pla.ne)*ee*pla.Te_eV/eps0)
    E_self_debye = @. sqrt(abs(ne_SM)*ee*pla.Te_eV/eps0)

    F.Epol_self = @. min(pla.γ_shape_fac * E_pol_required_for_cancellation, E_self_debye)
    extrapolate_field_to_boundary_nodes!(RP.G, F.Epol_self)

    calculate_parallel_electric_field!(RP)

    if RP.flags.mean_ExB
        @. pla.mean_ExB_R = (F.Epol_self / F.Btot) * sign(F.Eϕ) * F.bpol_Z
        @. pla.mean_ExB_Z = (F.Epol_self / F.Btot) * sign(F.Eϕ) * (-F.bpol_R)

        @. F.ER = -sign(F.Eϕ.*F.Bϕ)*F.Epol_self*F.bpol_R;
        @. F.EZ = -sign(F.Eϕ.*F.Bϕ)*F.Epol_self*F.bpol_Z;
    end

    # # Turbulent parallel diffusion based on fluctuation levels
    # # TODO: implement computing length of connection length
    # F.L_mixing .= 10.0;
    tp = RP.transport
    tp.L_mixing .= RP.flf.Lpol_tot

    extrapolate_field_to_boundary_nodes!(RP.G, tp.L_mixing)
    # smooth_data_2D!(tp.L_mixing; num_SM = 3)

    # # Turbulent diffusion coefficient
    # Dpol_turb = 0.5 * v_(ExB) * L_mixing
    @. tp.Dpol_turb = 0.5 * F.Epol_self / F.Btot * tp.L_mixing;

    return RP
end
