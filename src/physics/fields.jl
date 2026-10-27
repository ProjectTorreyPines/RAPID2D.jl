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
export update_prescribed_fields!,
    combine_field_sources!,
    calculate_magnetic_field_unit_vectors!,
    calculate_B_from_ψ!,
    calculate_B_from_ψ

# Export external field types and functions
export AbstractExternalField, TimeSeriesExternalField
export calculate_external_fields_at_time

# Required imports for field calculations
using LinearAlgebra

# Required imports for Green's function calculations


"""
    calculate_derived_magnetic_field_quantities!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate derived magnetic field quantities based on the current field values.
"""
function calculate_derived_magnetic_field_quantities!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @. RP.fields.Bpol = sqrt(RP.fields.BR^2 + RP.fields.BZ^2)
    @. RP.fields.Btot = sqrt(RP.fields.Bpol^2 + RP.fields.Bϕ^2)

    calculate_magnetic_field_unit_vectors!(RP)

    return RP
end

"""
    calculate_magnetic_field_unit_vectors!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Calculate unit vectors for the magnetic field.
"""
function calculate_magnetic_field_unit_vectors!(RP::RAPID{FT}) where {FT <: AbstractFloat}

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

Calculate the parallel electric field: the prescribed field's, the vertical-field control's,
and the self-generated ones.
"""
function calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    F = RP.fields
    # the prescribed field's parallel electric field, and the vertical-field control's
    control = _vertical_field_control_on(RP)
    @. F.Eϕ_para_presc = F.Eϕ_presc * F.bϕ
    if control
        @. F.Eϕ_para_ctrl = F.Eϕ_ctrl * F.bϕ
    end

    # the charge sheets' parallel field, against the applied loop voltage; none left from before
    # when the cancellation is off
    if RP.flags.E_para_cancellation
        if control
            @. F.ERZ_para_pla = sign(-(F.Eϕ_presc + F.Eϕ_ctrl) * F.Bϕ) * F.ERZ_pla_mag * F.Bpol / F.Btot
        else
            @. F.ERZ_para_pla = sign(-F.Eϕ_presc * F.Bϕ) * F.ERZ_pla_mag * F.Bpol / F.Btot
        end
    else
        fill!(F.ERZ_para_pla, zero(FT))
    end
    if RP.flags.E_para_induction
        @. F.Eϕ_para_coils_pla = F.Eϕ_coils_pla * F.bϕ
    end

    combine_E_para!(RP)
    return RP
end

# E∥ total: the prescribed field's, the vertical-field control's (only in a run that has one), and
# the self-generated ones
function combine_E_para!(RP::RAPID)
    F = RP.fields
    if _vertical_field_control_on(RP)
        @. F.E_para_tot = F.Eϕ_para_presc + F.Eϕ_para_ctrl + F.ERZ_para_pla + F.Eϕ_para_coils_pla
    else
        @. F.E_para_tot = F.Eϕ_para_presc + F.ERZ_para_pla + F.Eϕ_para_coils_pla
    end
    return RP
end

"""
    update_prescribed_fields!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}

Update the prescribed fields at the current simulation time or a specified time: interpolated
from the external field source (field files) when there is one, left as set by the manual setup
otherwise, and the toroidal electric field from the loop voltage.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `time_s::FT`: Time at which to update fields (default: current simulation time)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance
"""
function update_prescribed_fields!(RP::RAPID{FT}, time_s::FT = RP.time_s) where {FT <: AbstractFloat}
    F = RP.fields

    # the manual setup's field stays as set; a field-file source is interpolated in time
    if !isnothing(RP.external_field)
        # the source's fields at the specified time
        extF = calculate_external_fields_at_time(RP.external_field, time_s)

        # Update field components
        F.BR_presc .= extF.BR
        F.BZ_presc .= extF.BZ
        F.LV_presc .= extF.LV
        F.ψ_presc .= extF.psi
    end

    # Set toroidal magnetic field
    F.Bϕ = F.R0B0 ./ RP.G.R2D

    # Calculate toroidal electric field
    F.Eϕ_presc .= F.LV_presc ./ (2π * RP.G.R2D)

    return RP
end

"""
    combine_field_sources!(RP::RAPID{FT}, time_s::FT=RP.time_s) where {FT<:AbstractFloat}

Sum the fields of each source, prescribed, coils and plasma (+ vertical-field control), and
calculate derived field quantities. This function does NOT update the sources' fields themselves,
but assumes they are already updated separately.

# Arguments
- `RP::RAPID{FT}`: The RAPID simulation instance
- `time_s::FT`: Time parameter (used only for documentation, not actually used in calculations)

# Returns
- `RP::RAPID{FT}`: The updated RAPID instance with combined fields and derived quantities

# Note
The prescribed fields should be updated separately using `update_prescribed_fields!` before calling
this function, and the coils' and plasma's fields by the Ampère solve.
"""
function combine_field_sources!(RP::RAPID{FT}, time_s::FT = RP.time_s) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "combine_field_sources!" begin
        F = RP.fields

        # prescribed + coils and plasma
        @. F.BR = F.BR_presc + F.BR_coils_pla
        @. F.BZ = F.BZ_presc + F.BZ_coils_pla
        @. F.Eϕ = F.Eϕ_presc + F.Eϕ_coils_pla
        @. F.ψ = F.ψ_presc + F.ψ_coils_pla
        # the vertical-field control's vacuum field, only in a run that has one
        if _vertical_field_control_on(RP)
            F.BR .+= F.BR_ctrl
            F.BZ .+= F.BZ_ctrl
            F.Eϕ .+= F.Eϕ_ctrl
            F.ψ .+= F.ψ_ctrl
        end

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
mutable struct TimeSeriesExternalField{FT <: AbstractFloat} <: AbstractExternalField{FT}
    time_s::Vector{FT}    # Time points [s]

    # Field components
    BR::Array{FT, 3}       # Radial magnetic field [T]
    BZ::Array{FT, 3}       # Vertical magnetic field [T]
    psi::Array{FT, 3}      # Magnetic flux [Wb/rad]
    LV::Array{FT, 3}       # Loop voltage [V]

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

Interpolate field values at a specific time from time series data. Outside the time range
the nearest slice is held; a single slice is a static field.

# Arguments
- `field::TimeSeriesExternalField{FT}`: The time series external field data
- `time::FT`: The time at which to get field values

# Returns
- `NamedTuple`: Contains fields BR, BZ, LV, psi interpolated at the specified time
"""
function calculate_external_fields_at_time(extF::TimeSeriesExternalField{FT}, time::FT) where {FT <: AbstractFloat}
    # The slice at or before `time` and the one after it. Outside the range, and for a single
    # slice (a static field), both are the nearest slice and the weight is zero.
    t = extF.time_s
    n = length(t)
    i = clamp(searchsortedlast(t, time), 1, n)
    j = min(i + 1, n)
    t_weight = j == i ? zero(FT) : clamp((time - t[i]) / (t[j] - t[i]), zero(FT), one(FT))

    # Linear interpolation in time
    BR = (1 - t_weight) * extF.BR[:, :, i] + t_weight * extF.BR[:, :, j]
    BZ = (1 - t_weight) * extF.BZ[:, :, i] + t_weight * extF.BZ[:, :, j]
    psi = (1 - t_weight) * extF.psi[:, :, i] + t_weight * extF.psi[:, :, j]
    LV = (1 - t_weight) * extF.LV[:, :, i] + t_weight * extF.LV[:, :, j]

    return (
        BR = BR,
        BZ = BZ,
        LV = LV,
        psi = psi,
        time_s = time,
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
    update_space_charge_weights!(RP::RAPID{FT}) -> (ERZ_target, E_Debye) where {FT<:AbstractFloat}

The space charge's regime from the present state: the shape factor `γ_shape_fac` (the field's shape
alone, on closed lines too), the critical densities of the open-line charge sheets (infinite on a
closed line, which carries none) and the weights of the two closures that model what the code does
not resolve:
- `nc_para = ε₀ (γ ERZ_req)²/(e Tₑ)`, where the sheets' Debye field reaches the field that cancels
  the driven E∥; `nc_perp`, where their E×B drift outruns the parallel streaming projected on the
  RZ plane (Yoo et al. 2018);
- `turbulence_weight = min(1, √(n/nc_perp))`, the onset of the turbulent E×B mixing; zero without
  `charge_sheet_field` or `turb_ExB_mixing`;
- `ambipolar_weight`, the ambipolar coupling of the species' parallel transport, in [0, 1]: 1 on closed
  lines, `min(1, √(n/nc_para))` on open ones; zero without `ambipolar_coupling`. It splits the
  electron pressure drive between u∥ and the ambipolar diffusion ([`electron_pressure_shares`](@ref)).

Returns the two fields from which [`estimate_electrostatic_field_effects!`](@ref) builds the sheets'
field:
- `ERZ_target = γ ERZ_req`, the RZ field that cancels the driven E∥, shaped by γ;
- `E_Debye`, the Debye field, the most the local plasma holds.
"""
function update_space_charge_weights!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    F = RP.fields
    G = RP.G
    @unpack me, ee, eps0 = RP.config.constants
    closed = RP.flf.is_closed

    # =========================================================================
    # 1. Shape factor γ_shape_fac: the field's shape alone, on closed lines too
    # =========================================================================
    # A closed line has no sheets; that is the source's topology, not a value of γ. Without a
    # poloidal field there is no shape.
    Bpol_mean = mean(F.Bpol[G.nodes.in_wall_nids])
    # tan_θB = F.Bpol ./ F.Bϕ
    # tan_θB_norm = tan_θB ./ maximum(tan_θB[G.nodes.in_wall_nids])

    # # Calculate shape factor using sigmoid functions (MATLAB lines 554-556)
    # term1 = my_sigmf(FT(1e4) * Bpol_norm, FT(20.0), FT(0.4))
    # term2 = my_sigmf(tan_θB_norm, FT(5.0), FT(0.4))
    # pla.γ_shape_fac = @. term1 * term2

    # γ₀, the shape factor at the mean B_pol (`config.charge_sheet_shape_factor`)
    γ0 = RP.config.charge_sheet_shape_factor
    if Bpol_mean > 0
        @. pla.γ_shape_fac = min(FT(1.0), γ0 * (F.Bpol / Bpol_mean))
    else
        fill!(pla.γ_shape_fac, zero(FT))
    end

    # =========================================================================
    # 2. What the sheets would need, and what they can hold
    # =========================================================================
    # - ERZ_req = |E∥,drive| B/Bpol: the RZ field whose parallel part cancels the driven E∥
    # - ERZ_target = γ ERZ_req: shaped by γ; where Bpol vanishes, its limit γ₀ |E∥,drive| B/⟨Bpol⟩
    # - E_Debye = √(n e Tₑ/ε₀): the most the local plasma holds
    EM = RP.flags.E_para_induction
    E_para_drive = if _vertical_field_control_on(RP)
        @. F.Eϕ_para_presc + F.Eϕ_para_ctrl + EM * F.Eϕ_para_coils_pla
    else
        @. F.Eϕ_para_presc + EM * F.Eϕ_para_coils_pla
    end
    ERZ_target = if Bpol_mean > 0
        @. ifelse(
            F.Bpol > 0, pla.γ_shape_fac * (abs(E_para_drive) * (F.Btot / F.Bpol)),
            γ0 * abs(E_para_drive) * F.Btot / Bpol_mean
        )
    else
        zero(E_para_drive)
    end

    ne_SM = smooth_data_2D(pla.ne; num_SM = 2, weighting = RP.G.inVol2D)
    E_Debye = @. sqrt(max(ne_SM, zero(FT)) * ee * max(pla.Te_eV, zero(FT)) / eps0)

    # =========================================================================
    # 3. Critical densities of the open-line sheets
    # =========================================================================
    # E_Debye reaches ERZ_target at nc_para: none where the plasma is cold, any where nothing is needed
    @. pla.nc_para = ifelse(closed | (pla.Te_eV <= 0), FT(Inf), eps0 * ERZ_target^2 / (ee * pla.Te_eV))
    # Bpol/γ = max(Bpol, ⟨Bpol⟩/γ₀), finite where Bpol vanishes
    @. pla.nc_perp = ifelse(
        closed | (Bpol_mean <= 0), FT(Inf), eps0 / me * (F.Btot / F.Bϕ * max(F.Bpol, Bpol_mean / γ0))^2
    )

    # =========================================================================
    # 4. The closures' weights
    # =========================================================================
    # The sheets' E×B drift outruns the parallel streaming projected on the RZ plane above nc_perp
    # (Yoo et al. 2018): the fluctuating field mixes from there on
    if RP.flags.charge_sheet_field && RP.flags.turb_ExB_mixing
        @. pla.turbulence_weight = min(one(FT), sqrt(max(ne_SM, zero(FT)) / pla.nc_perp))
    else
        fill!(pla.turbulence_weight, zero(FT))
    end
    # The parallel pressure-driven flow of one species alone charges the plasma wherever its
    # divergence does not vanish, so the ambipolar field couples the species: always on a closed
    # line; on an open line as far as the sheets' charge reaches, f∥ = min(1, E_Debye/ERZ_target) — a
    # plasma that holds the sheets' charge holds the far smaller ambipolar charge
    if RP.flags.ambipolar_coupling
        @. pla.ambipolar_weight = ifelse(
            closed, one(FT),
            ifelse(E_Debye > 0, ifelse(ERZ_target > 0, min(one(FT), E_Debye / ERZ_target), one(FT)), zero(FT))
        )
    else
        fill!(pla.ambipolar_weight, zero(FT))
    end

    return ERZ_target, E_Debye
end

"""
    estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT<:AbstractFloat}

The open-line charge sheets' field and its effects on transport, from the present state (MATLAB
`Estimate_E_self_effects`, c_RAPID.m lines 544–679): the regime and the closures' weights
([`update_space_charge_weights!`](@ref)); with `charge_sheet_field`, the sheets' field
`ERZ_pla_mag = min(γ ERZ_req, E_Debye)`, zero on closed lines, which short-circuit along B; its parallel
projection `ERZ_para_pla` ([`calculate_parallel_electric_field!`](@ref)), the mean E×B drift and the
turbulent diffusivity `Dpol_turb` at the turbulence's weight. Without it, no field, parallel part
or diffusivity, so a run that turns it off keeps none from before.
"""
function estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    F = RP.fields
    G = RP.G
    ERZ_target, E_Debye = update_space_charge_weights!(RP)
    if !RP.flags.charge_sheet_field
        fill!(F.ERZ_pla_mag, zero(FT))
        fill!(RP.transport.Dpol_turb, zero(FT))
        # E∥ without the sheets' part at once: the transport's refresh reads it
        calculate_parallel_electric_field!(RP)
        return RP
    end

    # The open-line charge sheets' field (MATLAB lines 582-620): none on a closed line, before and
    # after the boundary nodes take their neighbours' mean
    @. F.ERZ_pla_mag = ifelse(RP.flf.is_closed, zero(FT), min(ERZ_target, E_Debye))
    extrapolate_field_to_boundary_nodes!(G, F.ERZ_pla_mag)
    F.ERZ_pla_mag[RP.flf.closed_surface_nids] .= zero(FT)

    calculate_parallel_electric_field!(RP)

    if RP.flags.mean_ExB
        @. pla.mean_ExB_R = (F.ERZ_pla_mag / F.Btot) * sign(F.Eϕ) * F.bpol_Z
        @. pla.mean_ExB_Z = (F.ERZ_pla_mag / F.Btot) * sign(F.Eϕ) * (-F.bpol_R)

        @. F.ER = -sign(F.Eϕ .* F.Bϕ) * F.ERZ_pla_mag * F.bpol_R
        @. F.EZ = -sign(F.Eϕ .* F.Bϕ) * F.ERZ_pla_mag * F.bpol_Z
    end

    # # Turbulent parallel diffusion based on fluctuation levels
    # # TODO: implement computing length of connection length
    # F.L_mixing .= 10.0;
    tp = RP.transport
    tp.L_mixing .= RP.flf.Lpol_tot

    extrapolate_field_to_boundary_nodes!(RP.G, tp.L_mixing)
    # smooth_data_2D!(tp.L_mixing; num_SM = 3)

    update_turbulent_diffusivity!(RP)

    return RP
end

"""
    update_turbulent_diffusivity!(RP) -> RP

The turbulent E×B mixing's diffusivity from the charge sheets' present field and the turbulence's
weight, `Dpol_turb = ½ w_turb (ERZ_pla_mag/B) L_mixing`: the drift speed `shared_turbulent_channel`
takes. Where the weights are refreshed without the field (a run's start, a failed step put back),
the diffusivity follows them.
"""
function update_turbulent_diffusivity!(RP::RAPID)
    pla, F, tp = RP.plasma, RP.fields, RP.transport
    @. tp.Dpol_turb = 0.5 * (pla.turbulence_weight * F.ERZ_pla_mag) / F.Btot * tp.L_mixing
    return RP
end

"""
    calculate_B_from_ψ!(G::GridGeometry{FT}, ψ::Matrix{FT}, BR::Matrix{FT}, BZ::Matrix{FT}) where {FT<:AbstractFloat}

Calculate magnetic field components (BR, BZ) from poloidal flux function ψ using in-place computation.

Uses the relations:
- BR = -(1/R)*(∂ψ/∂Z)
- BZ = (1/R)*(∂ψ/∂R)

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing R, Z coordinates and spacing
- `ψ::Matrix{FT}`: Poloidal flux function on the grid
- `BR::Matrix{FT}`: Pre-allocated matrix for radial magnetic field component (modified in-place)
- `BZ::Matrix{FT}`: Pre-allocated matrix for vertical magnetic field component (modified in-place)

# Notes
- Uses second-order central differences for interior points
- Uses first-order forward/backward differences at boundaries
- All matrices must have dimensions (G.NR, G.NZ)
"""
function calculate_B_from_ψ!(G::GridGeometry{FT}, ψ::Matrix{FT}, BR::Matrix{FT}, BZ::Matrix{FT}) where {FT <: AbstractFloat}
    @assert size(ψ) == (G.NR, G.NZ) "ψ must match grid size (NR=$(G.NR), NZ=$(G.NZ))"
    @assert size(BR) == (G.NR, G.NZ) "BR must match grid size (NR=$(G.NR), NZ=$(G.NZ))"
    @assert size(BZ) == (G.NR, G.NZ) "BZ must match grid size (NR=$(G.NR), NZ=$(G.NZ))"

    NR, NZ = G.NR, G.NZ

    inv_dR = one(FT) / G.dR
    inv_dZ = one(FT) / G.dZ
    half = FT(0.5)

    # BR = -(1/R)*(∂ψ/∂Z)
    # BZ = (1/R)*(∂ψ/∂R)

    # Interior points: use second-order central differences
    @inbounds for j in 2:(NZ - 1)
        for i in 2:(NR - 1)
            inv_R = one(FT) / G.R1D[i]
            BR[i, j] = inv_R * (- (ψ[i, j + 1] - ψ[i, j - 1]) * half * inv_dZ)
            BZ[i, j] = inv_R * ((ψ[i + 1, j] - ψ[i - 1, j]) * half * inv_dR)
        end
    end

    # R-direction boundaries (left and right edges)
    @inbounds for j in 1:NZ
        for i in [1, NR]
            inv_R = one(FT) / G.R1D[i]

            # BZ component: use forward/backward difference in R
            if i == 1
                # Left boundary: forward difference in R
                BZ[i, j] = inv_R * ((ψ[i + 1, j] - ψ[i, j]) * inv_dR)
            else  # i == NR
                # Right boundary: backward difference in R
                BZ[i, j] = inv_R * ((ψ[i, j] - ψ[i - 1, j]) * inv_dR)
            end

            # BR component: use appropriate difference in Z
            if j == 1
                # Bottom corner: forward difference in Z
                BR[i, j] = inv_R * (- (ψ[i, j + 1] - ψ[i, j]) * inv_dZ)
            elseif j == NZ
                # Top corner: backward difference in Z
                BR[i, j] = inv_R * (- (ψ[i, j] - ψ[i, j - 1]) * inv_dZ)
            else
                # Interior in Z: central difference in Z
                BR[i, j] = inv_R * (- (ψ[i, j + 1] - ψ[i, j - 1]) * half * inv_dZ)
            end
        end
    end

    # Z-direction boundaries (bottom and top edges, excluding corners already handled)
    return @inbounds for i in 2:(NR - 1)
        for j in [1, NZ]
            inv_R = one(FT) / G.R1D[i]

            # BR component: use forward/backward difference in Z
            if j == 1
                # Bottom boundary: forward difference in Z
                BR[i, j] = inv_R * (- (ψ[i, j + 1] - ψ[i, j]) * inv_dZ)
            else  # j == NZ
                # Top boundary: backward difference in Z
                BR[i, j] = inv_R * (- (ψ[i, j] - ψ[i, j - 1]) * inv_dZ)
            end

            # BZ component: central difference in R (interior points)
            BZ[i, j] = inv_R * ((ψ[i + 1, j] - ψ[i - 1, j]) * half * inv_dR)
        end
    end
end

"""
    calculate_B_from_ψ(G::GridGeometry{FT}, ψ::Matrix{FT}) where {FT<:AbstractFloat}

Calculate magnetic field components (BR, BZ) from poloidal flux function ψ.

This is a convenience wrapper around `calculate_B_from_ψ!` that allocates output arrays.

# Arguments
- `G::GridGeometry{FT}`: Grid geometry containing R, Z coordinates and spacing
- `ψ::Matrix{FT}`: Poloidal flux function on the grid

# Returns
- `Tuple{Matrix{FT}, Matrix{FT}}`: (BR, BZ) magnetic field components

# See also
- [`calculate_B_from_ψ!`](@ref): In-place version for better performance
"""
function calculate_B_from_ψ(G::GridGeometry{FT}, ψ::Matrix{FT}) where {FT <: AbstractFloat}
    @assert size(ψ) == (G.NR, G.NZ) "ψ must match grid size (NR=$(G.NR), NZ=$(G.NZ))"

    # Initialize BR and BZ matrices
    BR = zeros(FT, G.NR, G.NZ)
    BZ = zeros(FT, G.NR, G.NZ)

    # Calculate the magnetic field from the poloidal flux function
    calculate_B_from_ψ!(G, ψ, BR, BZ)

    return BR, BZ
end
