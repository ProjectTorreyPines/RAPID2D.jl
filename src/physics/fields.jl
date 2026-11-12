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
and the self-generated ones. The sheets' part is their mean poloidal field projected with the
sign that opposes the drive it answers ([`driven_parallel_field`](@ref), the same drive that
sizes it), so the projection, `ER`/`EZ` and the mean drift read one sign.
"""
function calculate_parallel_electric_field!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    F = RP.fields
    # the prescribed field's parallel electric field, the vertical-field control's and the induced
    control = _vertical_field_control_on(RP)
    @. F.Eϕ_para_presc = F.Eϕ_presc * F.bϕ
    if control
        @. F.Eϕ_para_ctrl = F.Eϕ_ctrl * F.bϕ
    end
    if RP.flags.E_para_induction
        @. F.Eϕ_para_coils_pla = F.Eϕ_coils_pla * F.bϕ
    end

    # the charge sheets' parallel field, against the drive; none left from before when the
    # cancellation is off
    if RP.flags.E_para_cancellation
        E_drive = driven_parallel_field(RP)
        @. F.Epol_para_pla = -sign(E_drive) * F.Epol_mean * F.Bpol / F.Btot
    else
        fill!(F.Epol_para_pla, zero(FT))
    end

    combine_E_para!(RP)
    return RP
end

# E∥ total: the prescribed field's, the vertical-field control's (only in a run that has one), and
# the self-generated ones
function combine_E_para!(RP::RAPID)
    F = RP.fields
    if _vertical_field_control_on(RP)
        @. F.E_para_tot = F.Eϕ_para_presc + F.Eϕ_para_ctrl + F.Epol_para_pla + F.Eϕ_para_coils_pla
    else
        @. F.E_para_tot = F.Eϕ_para_presc + F.Epol_para_pla + F.Eϕ_para_coils_pla
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
    update_space_charge_weights!(RP::RAPID{FT}) -> (; Epol_req, Epol_sat, E_Debye, E_para_drive) where {FT<:AbstractFloat}

The open-line charge sheets' three ceilings and the space charge's regime, from the present state.
The ceilings [V/m, magnitudes], which are returned:
- `Epol_req = |E∥,drive| B/B_pol`, the RZ field whose parallel part cancels the driven E∥
  ([`sheet_field_requirement`](@ref));
- `Epol_sat`, the saturation ceiling of `flags.mean_sheet_field` ([`mean_sheet_saturation`](@ref)): what
  the field's geometry lets the sheets sustain in the mean, however dense the plasma;
- `E_Debye = √(n e Tₑ/ε₀)`, the most the local plasma holds;
with `E_para_drive`, the driven parallel field ([`driven_parallel_field`](@ref)).
[`estimate_electrostatic_field_effects!`](@ref) takes the mean field as the least of the three.

The regime is read off one reference whatever the policy, the local ceiling
`Epol_local = min(Epol_req, γ₀ E₀)` with `E₀ = |E∥,drive| B/⟨B_pol⟩` the requirement at the in-wall
mean ⟨B_pol⟩ and γ₀ `config.charge_sheet_shape_factor`:
- `nc_para = ε₀ Epol_local²/(e Tₑ)`, the density at which `E_Debye` reaches `Epol_local`; `nc_perp`,
  where the sheets' E×B drift outruns the parallel streaming projected on the RZ plane (Yoo et al.
  2018). Both infinite on a closed line, which carries no sheets;
- `turbulence_weight = min(1, √(n/nc_perp))`, the onset of the turbulent E×B mixing; zero without
  `charge_sheet_field` or `turb_ExB_mixing`;
- `ambipolar_weight`, the ambipolar coupling of the species' parallel transport, in [0, 1]: 1 on closed
  lines, `min(1, E_Debye/Epol_local) = min(1, √(n/nc_para))` on open ones; zero without
  `ambipolar_coupling`. It splits the electron pressure drive between u∥ and the ambipolar diffusion
  ([`electron_pressure_shares`](@ref)).
"""
function update_space_charge_weights!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    F = RP.fields
    G = RP.G
    @unpack me, ee, eps0 = RP.config.constants
    closed = RP.flf.is_closed
    γ0 = RP.config.charge_sheet_shape_factor
    Bpol_in_wall_avg = mean(F.Bpol[G.nodes.in_wall_nids])

    # =========================================================================
    # 1. The three ceilings of the sheets' mean field
    # =========================================================================
    E_para_drive = driven_parallel_field(RP)
    Epol_req = sheet_field_requirement(RP, E_para_drive)
    # E₀, the requirement at the in-wall mean B_pol (none without a poloidal field), and the local ceiling γ₀ E₀
    Epol_ref = Bpol_in_wall_avg > 0 ? (@. abs(E_para_drive) * F.Btot / Bpol_in_wall_avg) : zero(E_para_drive)
    Epol_sat_local = γ0 .* Epol_ref
    ne_SM = smooth_data_2D(pla.ne; num_SM = 2, weighting = RP.G.inVol2D)
    E_Debye = debye_field(RP, ne_SM)

    # =========================================================================
    # 2. Critical densities of the open-line sheets, from the local ceiling
    # =========================================================================
    Epol_local = @. min(Epol_req, Epol_sat_local)
    # E_Debye reaches Epol_local at nc_para: none where the plasma is cold, any where nothing is needed
    @. pla.nc_para = ifelse(closed | (pla.Te_eV <= 0), FT(Inf), eps0 * Epol_local^2 / (ee * pla.Te_eV))
    # |E∥,drive| B/Epol_local = max(Bpol, ⟨Bpol⟩/γ₀), written by the field's geometry so that it needs no drive
    @. pla.nc_perp = ifelse(
        closed | (Bpol_in_wall_avg <= 0), FT(Inf), eps0 / me * (F.Btot / F.Bϕ * max(F.Bpol, Bpol_in_wall_avg / γ0))^2
    )

    # =========================================================================
    # 3. The closures' weights
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
    # line; on an open line as far as the sheets' charge reaches, min(1, E_Debye/Epol_local) — a
    # plasma that holds the sheets' charge holds the far smaller ambipolar charge
    if RP.flags.ambipolar_coupling
        @. pla.ambipolar_weight = ifelse(
            closed, one(FT),
            ifelse(E_Debye > 0, ifelse(Epol_local > 0, min(one(FT), E_Debye / Epol_local), one(FT)), zero(FT))
        )
    else
        fill!(pla.ambipolar_weight, zero(FT))
    end

    # =========================================================================
    # 4. The mean field's saturation ceiling: the local one unless the policy says otherwise
    # =========================================================================
    Epol_sat = mean_sheet_saturation(RP.flags.mean_sheet_field, RP, Epol_ref, Epol_sat_local, Bpol_in_wall_avg)

    return (; Epol_req, Epol_sat, E_Debye, E_para_drive)
end

"""
    sheet_field_requirement(RP, E_para_drive) -> Matrix

`Epol_req = |E∥,drive| B/B_pol` [V/m]: the RZ field along `b̂_pol` whose parallel part cancels the
driven E∥. Where B_pol vanishes no finite field does (`Inf`); nothing where nothing is driven.
"""
function sheet_field_requirement(RP::RAPID{FT}, E_para_drive) where {FT}
    F = RP.fields
    return @. ifelse(
        F.Bpol > 0, abs(E_para_drive) * (F.Btot / F.Bpol), ifelse(abs(E_para_drive) > 0, FT(Inf), zero(FT))
    )
end

"""
    mean_sheet_saturation(policy, RP, Epol_ref, Epol_sat_local, Bpol_in_wall_avg) -> Matrix

The saturation ceiling `Epol_sat` of the sheets' mean field [V/m, a magnitude] under the
[`MeanSheetFieldPolicy`](@ref) `policy`. `Epol_ref` is `E₀ = |E∥,drive| B/⟨B_pol⟩`, the requirement at
the in-wall mean B_pol `Bpol_in_wall_avg` (zero without a poloidal field, and so is every ceiling); `Epol_sat_local` the
local ceiling `γ₀ E₀`, which the default returns as it is.
"""
mean_sheet_saturation(::LocalPitch, RP::RAPID, Epol_ref, Epol_sat_local, Bpol_in_wall_avg) = Epol_sat_local

function mean_sheet_saturation(::LineMinPitch, RP::RAPID{FT}, Epol_ref, Epol_sat_local, Bpol_in_wall_avg) where {FT}
    F, flf = RP.fields, RP.flf
    # the local ceiling lowered by the least B_pol met along the node's line against the node's own; as it
    # is where no line was followed (outside the wall) and where B_pol vanishes
    return @. Epol_sat_local * ifelse((F.Bpol > 0) & isfinite(flf.min_Bpol), flf.min_Bpol / F.Bpol, one(FT))
end

function mean_sheet_saturation(policy::LineAvgSaturation, RP::RAPID{FT}, Epol_ref, Epol_sat_local, Bpol_in_wall_avg) where {FT}
    F, flf = RP.fields, RP.flf
    Epol_sat = zero(Epol_ref)
    Bpol_in_wall_avg > 0 || return Epol_sat
    p, coefficient = FT(policy.p), FT(policy.coefficient)
    @with_pool pool begin
        # the line's effective B_pol: ⟨B_pol⁻ᵖ⟩^(-1/p) from the following's second integral, or the mean of B/B_pol
        # its lengths hold (p = 1) when the following did not carry it
        Bpol_line = acquire!(pool, FT, size(Epol_sat)...)
        if flf.has_second_integral
            @. Bpol_line = (flf.second_integral_tot / flf.Lpol_tot)^(-1 / p)
        else
            @. Bpol_line = F.Btot * flf.Lpol_tot / flf.Lc_tot
        end
        # a line that was not measured: one that ends on a null (no RZ length to average over) has none, a node
        # no line was followed from keeps its own
        @. Bpol_line = ifelse(
            isfinite(Bpol_line) & (Bpol_line > 0), Bpol_line, ifelse(isfinite(flf.Lpol_tot), F.Bpol, zero(FT))
        )
        @. Epol_sat = coefficient * Epol_ref * (Bpol_line / (Bpol_line + Bpol_in_wall_avg))
    end
    return Epol_sat
end

"""
    mean_sheet_line_integrand(RP) -> Union{Nothing, Matrix}

The node field the mean field's policy has the field-line following integrate along every line
(`second_integrand`, `flf.second_integral_tot`): `B_pol⁻ᵖ` for [`LineAvgSaturation`](@ref), nothing for
the others. Infinite on a node where B_pol vanishes, which the following's linear interpolation leaves
out of the cells around it.
"""
mean_sheet_line_integrand(RP::RAPID) = mean_sheet_line_integrand(RP.flags.mean_sheet_field, RP)
mean_sheet_line_integrand(::MeanSheetFieldPolicy, RP::RAPID) = nothing
mean_sheet_line_integrand(policy::LineAvgSaturation, RP::RAPID) = RP.fields.Bpol .^ (-eltype(RP.fields.Bpol)(policy.p))


"""
    driven_parallel_field(RP) -> Matrix

The parallel field the charge sheets answer: the prescribed, the control's (when the vertical-field
control is on) and, with `E_para_induction`, the induced parallel field.
"""
function driven_parallel_field(RP::RAPID)
    F = RP.fields
    EM = RP.flags.E_para_induction
    return if _vertical_field_control_on(RP)
        @. F.Eϕ_para_presc + F.Eϕ_para_ctrl + EM * F.Eϕ_para_coils_pla
    else
        @. F.Eϕ_para_presc + EM * F.Eϕ_para_coils_pla
    end
end

# The Debye field √(n e Tₑ/ε₀) [V/m] of the density `ne` (the weights' smoothed one), the most the
# local plasma holds; zero where it is empty or cold
function debye_field(RP::RAPID{FT}, ne) where {FT}
    @unpack ee, eps0 = RP.config.constants
    return @. sqrt(max(ne, zero(FT)) * ee * max(RP.plasma.Te_eV, zero(FT)) / eps0)
end

"""
    sheet_mixing_integrand(RP) -> Matrix

The field the mixing amplitude averages along each line, at the nodes [V/m]: the RZ field that would
cancel the driven E∥ there, `|E∥,drive| B/B_pol`, capped at what the plasma holds, `E_Debye` — the same
cap as the mean field's, applied before the line average so that no cell contributes more than
`E_Debye Δx` of potential. The field-line following integrates it (`field_line_analysis!`'s
`integrand`, `flf.integral_tot`); the step loop passes it at every refresh, the setup (before the plasma
is set) none. Where B_pol vanishes the cap alone remains; an empty or cold cell gives zero.
"""
function sheet_mixing_integrand(RP::RAPID{FT}) where {FT}
    ne_SM = smooth_data_2D(RP.plasma.ne; num_SM = 2, weighting = RP.G.inVol2D)
    E_Debye = debye_field(RP, ne_SM)
    return min.(sheet_field_requirement(RP, driven_parallel_field(RP)), E_Debye)
end

"""
    estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT<:AbstractFloat}

The open-line charge sheets' two fields and their effects on transport, from the present state
(MATLAB `Estimate_E_self_effects`, c_RAPID.m lines 544–679, which had one): the regime and the
closures' weights ([`update_space_charge_weights!`](@ref)); with `charge_sheet_field`,
by component — "pol" along `b̂_pol` (the poloidal direction), "rad" along the flux-normal `∇ψ/|∇ψ| = φ̂ × b̂_pol` (across the lines) —
- the **mean poloidal component**, `Epol_mean = min(Epol_req, Epol_sat, E_Debye)` (Ē_pol, a magnitude;
  the sign is set where it is applied): the time-averaged field the sheets sustain, the least of the
  requirement, the saturation ceiling of `flags.mean_sheet_field` ([`MeanSheetFieldPolicy`](@ref)) and
  what the plasma holds. It is the one
  component that projects onto E∥ — `Epol_para_pla` cancels the driven E∥
  ([`calculate_parallel_electric_field!`](@ref)) — and its E×B drift is the mean drift **across** the lines;
- the **fluctuating flux-normal component**, `Erad_fluc = min(⟨min(|E∥,drive| B/B_pol, E_Debye)⟩_line, E_Debye)`
  (δE_r): the cancelling field capped cell by cell at what the plasma holds, averaged along the line over
  its RZ length (`flf.integral_tot/Lpol_tot`, [`sheet_mixing_integrand`](@ref); before the step loop's
  first following, the uncapped `|E∥| L_c/L_pol`). An effective amplitude, not an rms; its E×B drift is
  **along** `b̂_pol`, the turbulent mixing — `Dpol_turb` ([`update_turbulent_diffusivity!`](@ref)) and
  `shared_turbulent_channel` (the electron wall, the ion channels and their wall) read it, nothing
  else. The saturation ceiling does not enter it: how much of the drive the mean field cancels and how hard the sheets
  fluctuate are separate questions. The other two components, a fluctuating one along `b̂_pol`
  (a fluctuating E∥) and a mean one across the lines (a mean flow along `b̂_pol`), are not modelled.

Both are zero on closed lines, which short-circuit along B, and take the same boundary extrapolation.
Without `charge_sheet_field` no field, parallel part or diffusivity, so a run that turns it off keeps
none from before; the mean drift `mean_ExB_R/Z` and `ER`/`EZ` are then not written at all — whatever
is set stands, a prescribed drift.
"""
function estimate_electrostatic_field_effects!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    F = RP.fields
    G = RP.G
    (; Epol_req, Epol_sat, E_Debye, E_para_drive) = update_space_charge_weights!(RP)
    if !RP.flags.charge_sheet_field
        fill!(F.Epol_mean, zero(FT))
        fill!(F.Erad_fluc, zero(FT))
        fill!(RP.transport.Dpol_turb, zero(FT))
        # `mean_ExB_R/Z` and `ER/EZ` are left as they are: without the sheets' field they are an
        # input, a prescribed drift (the wall and convection tests set one)
        # E∥ without the sheets' part at once: the transport's refresh reads it
        calculate_parallel_electric_field!(RP)
        return RP
    end

    # The open-line charge sheets' field (MATLAB lines 582-620): none on a closed line, before and
    # after the boundary nodes take their neighbours' mean
    @. F.Epol_mean = ifelse(RP.flf.is_closed, zero(FT), min(Epol_req, Epol_sat, E_Debye))
    extrapolate_field_to_boundary_nodes!(G, F.Epol_mean)
    F.Epol_mean[RP.flf.closed_surface_nids] .= zero(FT)

    # The mixing amplitude: the capped cancelling field averaged along the line, the local cap again
    # (the line's mean must be held here too); before the first following with the integrand (the
    # setup's has no plasma yet) the uncapped line average |E∥| L_c/L_pol; nothing where the line was
    # not measured or there is no poloidal field (a null's line is toroidal: L_pol = Inf gives 0 too)
    flf = RP.flf
    line_avg = flf.has_integral ? flf.integral_tot ./ flf.Lpol_tot : abs.(E_para_drive) .* flf.Lc_tot ./ flf.Lpol_tot
    @. F.Erad_fluc = ifelse(
        flf.is_closed | !(F.Bpol > 0) | !(flf.Lpol_tot > 0) | !isfinite(line_avg), zero(FT),
        min(line_avg, E_Debye)
    )
    extrapolate_field_to_boundary_nodes!(G, F.Erad_fluc)
    F.Erad_fluc[flf.closed_surface_nids] .= zero(FT)

    calculate_parallel_electric_field!(RP)

    if RP.flags.mean_ExB
        # the mean field is s Ē_pol b̂_pol with s = −sgn(E∥,drive), the sign that cancels the drive it
        # answers (the one `Epol_para_pla` takes); its drift s Ē_pol b̂_pol × B_ϕ φ̂ / B² runs along
        # −s sgn(B_ϕ) (b_Z, −b_R): across the lines, toward the downstream wall
        s_drive = @. -sign(E_para_drive)
        @. pla.mean_ExB_R = (F.Epol_mean / F.Btot) * (-(s_drive * sign(F.Bϕ))) * F.bpol_Z
        @. pla.mean_ExB_Z = (F.Epol_mean / F.Btot) * (s_drive * sign(F.Bϕ)) * F.bpol_R

        @. F.ER = s_drive * F.Epol_mean * F.bpol_R
        @. F.EZ = s_drive * F.Epol_mean * F.bpol_Z
    end

    # # Turbulent parallel diffusion based on fluctuation levels
    # # TODO: implement computing length of connection length
    # F.L_mixing .= 10.0;
    # the mixing length: the line's RZ length; a null's line (L_pol = Inf, toroidal) mixes along nothing
    tp = RP.transport
    @. tp.L_mixing = ifelse(isfinite(flf.Lpol_tot), flf.Lpol_tot, zero(FT))

    extrapolate_field_to_boundary_nodes!(RP.G, tp.L_mixing)
    # smooth_data_2D!(tp.L_mixing; num_SM = 3)

    update_turbulent_diffusivity!(RP)

    return RP
end

"""
    update_turbulent_diffusivity!(RP) -> RP

The turbulent E×B mixing's diffusivity along `b̂_pol` from the charge sheets' fluctuating field across
the lines and the turbulence's weight, `Dpol_turb = ½ w_turb (Erad_fluc/B) L_mixing`: the drift speed
`shared_turbulent_channel` takes, and the one entry of the tensor `Dpol_turb b̂_pol b̂_pol`
(`update_diffusion_tensor!`). With `L_mixing = L_pol,tot` the length cancels:
`Dpol_turb = ½ w_turb ∫min(|E∥|B/B_pol, E_Debye)dl_pol/B` up to the outer cap, one value along a line.
Where the weights are refreshed without the field (a run's start, a failed step put back), the
diffusivity follows them.
"""
function update_turbulent_diffusivity!(RP::RAPID)
    pla, F, tp = RP.plasma, RP.fields, RP.transport
    @. tp.Dpol_turb = 0.5 * (pla.turbulence_weight * F.Erad_fluc) / F.Btot * tp.L_mixing
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
