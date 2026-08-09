using HDF5
using FastInterpolations

"""
    AbstractReactionRateCoefficient{T<:AbstractFloat}

Abstract type for all reaction rate coefficient models.
Concrete implementations must provide methods to compute or interpolate reaction rates for different conditions.
"""
abstract type AbstractReactionRateCoefficient{T <: AbstractFloat} end

"""
    RRC_EoverP_Erg{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on electric field over pressure (E/p) and particle energy.
Stores both raw data and an interpolation object for efficient calculation of rate coefficients.

# Fields
- `EoverP::Vector{FT}`: Electric field over pressure (E/p) coordinates
- `Erg_eV::Vector{FT}`: Particle energy in eV
- `raw_data::AbstractArray{FT}`: Raw reaction rate data as a matrix
- `itp`: Interpolant; clamps to the table boundary outside its bounds
- `dK_dĒ`: `∂K/∂Ē` at the same point — see below

# The derivative surface

`dK_dĒ` is the analytic `∂K/∂Ē` of the same bilinear interpolant, with the same
query shapes as `itp`. `∂Ē/∂Tₑ = 3/2` is constant, so the chain rule to `∂P/∂Tₑ`
is one scalar multiply and no finite differencing.

**Its extrapolation differs from `itp`'s per axis, and the asymmetry is
load-bearing.** `E/p` clamps as `itp` does — it carries no `Tₑ` dependence. `Ē`
uses `NoExtrap` and **callers clamp it themselves**: out of range the value is
frozen so the true derivative is `0`, but FastInterpolations' `ClampExtrap`
returns the boundary cell's slope (upstream bug, derivative views only).
[`update_rate_jacobian!`](@ref) clamps and masks; `NoExtrap` stands behind it as
an assertion. Out of range is ordinary — `apply_electron_density_boundary_conditions!`
damps `Tₑ` toward zero outside the wall, putting those nodes at `Ē = 0`.

When the upstream fix lands, drop the second interpolant and take
`deriv_view(itp, (0, 1))` directly."""
struct RRC_EoverP_Erg{FT <: AbstractFloat} <: AbstractReactionRateCoefficient{FT}
    # 2 variables for given reaction rate coefficient
    EoverP::Vector{FT}  # Electric field over pressure (E/p) coordinates
    Erg_eV::Vector{FT}  # Particle's energy

    raw_data::AbstractArray{FT}
    itp  # Interpolant; clamps to the table boundary outside its bounds
    dK_dĒ  # ∂K/∂Ē of the same interpolant; clamps on E/p, raises on Ē (see above)

    function RRC_EoverP_Erg(EoverP::Vector{FT}, Erg_eV::Vector{FT}, raw_data::AbstractArray{FT}) where {FT <: AbstractFloat}
        # ClampExtrap: below the table's minimum E/p the rate relaxes to the room-T
        # Maxwellian (bottom row), not 0 — E/p=0 means no field, not no collisions.
        itp = linear_interp((EoverP, Erg_eV), raw_data; extrap = ClampExtrap())
        itp_d = linear_interp(
            (EoverP, Erg_eV), raw_data;
            extrap = (ClampExtrap(), NoExtrap())
        )
        return new{FT}(EoverP, Erg_eV, raw_data, itp, deriv_view(itp_d, (0, 1)))
    end
end

"""
    RRC_T_ud{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on temperature and parallel drift velocity.
Used for reactions where the rate depends on temperature and drift velocity.

# Fields
- `T_eV::Vector{FT}`: Temperature in eV
- `ud_para::Vector{FT}`: Parallel drift velocity
- `raw_data::AbstractArray{FT}`: Raw reaction rate data as a matrix
- `itp`: Interpolant; clamps to the table boundary outside its bounds
- `dK_dT`: `∂K/∂T` at the same point

# The derivative surface

Mirror of [`RRC_EoverP_Erg`](@ref)'s, and simpler: temperature is this table's
own first axis, so there is no chain rule at all — `∂ν/∂T = n_gas·∂K/∂T`.

Same per-axis asymmetry, for the same reason. `u_d` clamps as `itp` does; `T`
uses `NoExtrap` and callers clamp it themselves, because out of range the value
is frozen and the honest derivative is `0`, not the boundary cell's slope
(FastInterpolations returns the latter — upstream bug, derivative views only).
"""
struct RRC_T_ud{FT <: AbstractFloat} <: AbstractReactionRateCoefficient{FT}
    # 2 variables for given reaction rate coefficient
    T_eV::Vector{FT}  # Temperature in eV
    ud_para::Vector{FT}  # parallel velocity

    raw_data::AbstractArray{FT}
    itp  # Interpolant; clamps to the table boundary outside its bounds
    dK_dT  # ∂K/∂T of the same interpolant; raises on T, clamps on u_d (see above)

    function RRC_T_ud(T_eV::Vector{FT}, ud_para::Vector{FT}, raw_data::AbstractArray{FT}) where {FT <: AbstractFloat}
        size(raw_data) == (length(T_eV), length(ud_para)) || throw(
            DimensionMismatch(
                "surface is $(size(raw_data)) but the axes are " *
                    "($(length(T_eV)) T_eV, $(length(ud_para)) ud_para). Axis 1 is " *
                    "TEMPERATURE — see T_ud_axis_order."
            )
        )
        # ClampExtrap: out-of-domain (T, u_d) queries clamp to the nearest boundary rate.
        itp = linear_interp((T_eV, ud_para), raw_data; extrap = ClampExtrap())
        itp_d = linear_interp(
            (T_eV, ud_para), raw_data;
            extrap = (NoExtrap(), ClampExtrap())
        )
        return new{FT}(T_eV, ud_para, raw_data, itp, deriv_view(itp_d, (1, 0)))
    end
end

"""
    T_ud_axis_order(h5fid) -> Symbol
    read_T_ud_surface(h5fid, name, order) -> Matrix

Read one `(T, u_d)` surface in [`RRC_T_ud`](@ref)'s own axis order, whichever way the
file stores it.

`RRC_T_ud(T_eV, ud_para, A)` means `A[i, j]` is the rate at `T_eV[i]`, `ud_para[j]`.
**The shipped files store the transpose** — axis 1 = drift, axis 2 = temperature — and
because both axes have the same length (100×100, 200×200) nothing raised for as long as
they have existed. It returned the right number at the wrong point.

The evidence, and why the files rather than the reader are the odd one out, is in
`internal/docs/src/notes/issues/ion-rrc-table-transposed.md`. In one line: the electron
`Ionization` surface read the constructor's way is 3.6e-14 m³/s at `T = 19` meV, which a
15.4 eV threshold forbids; read the other way it is zero below 1.7 eV and rises through
the tens of eV.

The layout is therefore a property of the file, declared by an `axis_order` attribute:

| `axis_order` | meaning |
|---|---|
| `"T_eV,ud_para"` | already the constructor's order — read as-is |
| `"ud_para,T_eV"` | the legacy layout — `permutedims` on read |
| absent | legacy. On this repository's history, an unstamped file is a transposed one |

**Transposing on read rather than rewriting the files is deliberate.** They are 2.6 MB of
binary, git keeps every version, and the ion cross-section overhaul will regenerate them
anyway — at which point it writes them in the constructor's order, stamps them, and
nothing here changes. Paying the blob twice to be correct in between buys nothing.
"""
function T_ud_axis_order(h5fid)
    haskey(attrs(h5fid), "axis_order") || return :transposed
    order = read_attribute(h5fid, "axis_order")
    order == "T_eV,ud_para" && return :as_stored
    order == "ud_para,T_eV" && return :transposed
    throw(
        ArgumentError(
            "unknown axis_order \"$order\": expected \"T_eV,ud_para\" or \"ud_para,T_eV\""
        )
    )
end

function read_T_ud_surface(h5fid, name::AbstractString, order::Symbol)
    A = read(h5fid, name)
    return order === :as_stored ? A : permutedims(A)
end

"""
    ion_rate_jacobian(RP, reaction) -> Matrix

`∂K/∂T_i` for one `(T, u_d)` ion surface, at the same `(T_i, |u_i∥|)` the value
path uses, with the out-of-range mask [`update_rate_jacobian!`](@ref) applies for
the electron surfaces and for the same reason.

Returns `∂K/∂T`, not `∂ν/∂T`: the caller owns the `n_gas` and the per-channel
weights (`½` on elastic), exactly as it owns them for the value.
"""
function ion_rate_jacobian(RP::RAPID{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    rrc = getfield(RP.iRRCs, reaction)
    rrc isa RRC_T_ud ||
        throw(ArgumentError("∂/∂T_i is defined for (T, u_d) surfaces; $reaction is not one"))

    T_lo, T_hi = first(rrc.T_eV), last(rrc.T_eV)
    T_query = clamp.(RP.plasma.Ti_eV, T_lo, T_hi)
    out = rrc.dK_dT((T_query, abs.(RP.plasma.ui_para)))
    # Exact equality is the in-range test — `clamp` returns its argument untouched
    # inside the interval and a bound outside it.
    @. out = ifelse(RP.plasma.Ti_eV == T_query, out, zero(FT))
    return out
end

"""
    RRC_T_ud_gFac{FT<:AbstractFloat} <: AbstractReactionRateCoefficient{FT}

Reaction rate coefficient model based on temperature, parallel drift velocity, and distribution function g-factor.
Used for more complex reactions where the distribution function shape affects the rate.

# Fields
- `T_eV::Vector{FT}`: Temperature in eV
- `ud_para::Vector{FT}`: Parallel drift velocity
- `gFac::Vector{FT}`: g-factor of the distribution function
- `raw_data::AbstractArray{FT}`: Raw reaction rate data
- `itp`: Interpolant; clamps to the table boundary outside its bounds
"""
struct RRC_T_ud_gFac{FT <: AbstractFloat} <: AbstractReactionRateCoefficient{FT}
    # 3 variables for given reaction rate coefficient
    T_eV::Vector{FT}  # Temperature in eV
    ud_para::Vector{FT}  # parallel velocity
    gFac::Vector{FT}  # g-factor of Distribution function

    raw_data::AbstractArray{FT}
    itp  # Interpolant; clamps to the table boundary outside its bounds

    function RRC_T_ud_gFac(T_eV::Vector{FT}, ud_para::Vector{FT}, gFac::Vector{FT}, raw_data::AbstractArray{FT}) where {FT <: AbstractFloat}
        # ClampExtrap: out-of-domain (T, u_d, gFac) queries clamp to the nearest boundary rate.
        itp = linear_interp((T_eV, ud_para, gFac), raw_data; extrap = ClampExtrap())
        return new{FT}(T_eV, ud_para, gFac, raw_data, itp)
    end
end

"""
    Electron_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}

Container for electron-related reaction rate coefficient models.
Stores various reaction models for electron-neutral and electron-ion interactions.

# Fields
- `Ionization`: Rate coefficient for electron impact ionization
- `Total_Momentum`: Drift-friction rate coefficient — the SUM over all electron-neutral
  channels (elastic + excitation + ionization) — the v_z-weighted moment
  ⟨σ_mom·|v|·v_z⟩/⟨v_z⟩, NOT the density-weighted collision frequency ⟨σ_mom·|v|⟩
- `Momentum_by_ela`: the ELASTIC share of `Total_Momentum`. Elastic energy transfer to the
  neutrals is ~2mₑ/M per *elastic* momentum-transfer collision only; the inelastic
  share of `Total_Momentum` carries its momentum into excitation/ionization, which are
  accounted separately. Using `Total_Momentum` for that term therefore over-counts it
  (~1.4× at E/p ≈ 100, ~4.4× at E/p ≈ 1000). Required — tables predating the
  per-channel split are not supported; regenerate with BreakdownDynamics'
  `postprocessing/make_envelope_h5.jl`.
- `Total_Excitation`: Energy-normalized excitation rate coefficient
  K_exc·ε_exc,eff/ε_ch, consumed as P_exc = ν_exc·ε_ch (see `characteristic_exc_erg_eV`)
- `Dissoc_Ionz`: Rate coefficient for dissociative ionization
- `Halpha`: Rate coefficient for Halpha emission
- `Recomb_H2Ion`: Rate coefficient for H2+ recombination
- `Recomb_H3Ion`: Rate coefficient for H3+ recombination
- `characteristic_exc_erg_eV`: the excitation normalization the loaded table was built
  with (`nothing` if the table omits it); validated in `initialize_RRCs!`.
"""
struct Electron_RRCs{FT <: AbstractFloat} <: AbstractSpeciesRRCs{FT}
    Ionization::RRC_EoverP_Erg{FT}
    Total_Momentum::RRC_EoverP_Erg{FT}
    Momentum_by_ela::RRC_EoverP_Erg{FT}
    Total_Excitation::RRC_EoverP_Erg{FT}

    Dissoc_Ionz::RRC_T_ud{FT}
    Halpha::RRC_T_ud{FT}
    Recomb_H2Ion::RRC_T_ud{FT}
    Recomb_H3Ion::RRC_T_ud{FT}

    characteristic_exc_erg_eV::Union{FT, Nothing}  # table's exc normalization; checked in initialize_RRCs!

    function Electron_RRCs(eRRC_EoverP_Erg_fileName::String, eRRC_T_ud_fileName::String)
        @assert isfile(eRRC_EoverP_Erg_fileName) "File not found: $eRRC_EoverP_Erg_fileName"
        @assert isfile(eRRC_T_ud_fileName) "File not found: $eRRC_T_ud_fileName"

        # Create RRC_EoverP_Erg objects for each reaction type from the given H5 file
        h5fid = h5open(eRRC_EoverP_Erg_fileName, "r")
        EoverP = read(h5fid, "EoverP")
        Erg_eV = read(h5fid, "Erg_eV")
        Ionization = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Ionization"))
        Total_Momentum = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Total_Momentum"))
        Momentum_by_ela = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Momentum_by_ela"))
        Total_Excitation = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, "Total_Excitation"))
        # Excitation normalization the table was built with — kept as a field and
        # validated later in initialize_RRCs! (where RP.config is available) against
        # config.constants.char_exc_erg_eV. `nothing` if the table omits it.
        char_exc = haskey(h5fid, "characteristic_exc_erg_eV") ?
            Float64(read(h5fid, "characteristic_exc_erg_eV")) : nothing
        close(h5fid)

        # Create RRC_T_ud objects for each reaction type from the given H5 file
        h5fid = h5open(eRRC_T_ud_fileName, "r")
        order = T_ud_axis_order(h5fid)
        T_eV = read(h5fid, "T_eV")
        ud_para = read(h5fid, "ud_para")
        srf(name) = read_T_ud_surface(h5fid, name, order)
        Dissoc_Ionz = RRC_T_ud(T_eV, ud_para, srf("Dissoc_Ionz"))
        Halpha = RRC_T_ud(T_eV, ud_para, srf("Halpha"))
        Recomb_H2Ion = RRC_T_ud(T_eV, ud_para, srf("Recomb_H2Ion"))
        Recomb_H3Ion = RRC_T_ud(T_eV, ud_para, srf("Recomb_H3Ion"))
        close(h5fid)


        FT = eltype(EoverP)  # Determine the floating-point type from the data

        return new{FT}(
            Ionization, Total_Momentum, Momentum_by_ela, Total_Excitation,
            Dissoc_Ionz, Halpha, Recomb_H2Ion, Recomb_H3Ion,
            char_exc === nothing ? nothing : FT(char_exc)
        )
    end
end

"""
    check_exc_erg_consistency(eRRCs::Electron_RRCs, char_exc_erg_eV)

Verify the loaded electron RRC table's excitation normalization matches RAPID2D's
`char_exc_erg_eV` (`config.constants`). The `Total_Excitation` surface is energy-normalized
to the table's `characteristic_exc_erg_eV`, so `P_exc = e·char_exc_erg_eV·n_gas·RRC` only
reproduces the kinetic loss if the two agree. Missing (`nothing`) → warn + assume our
value; present but different → error. Called from `initialize_RRCs!`.
"""
function check_exc_erg_consistency(eRRCs::Electron_RRCs, char_exc_erg_eV::Real)
    ch = eRRCs.characteristic_exc_erg_eV
    if ch === nothing
        @warn "Electron RRC table has no characteristic_exc_erg_eV; " *
            "assuming $char_exc_erg_eV eV (RAPID2D's char_exc_erg_eV)."
    else
        isapprox(ch, char_exc_erg_eV; rtol = 1.0e-6) || error(
            "Electron RRC table is normalized to characteristic_exc_erg_eV = $ch eV, " *
                "but RAPID2D uses char_exc_erg_eV = $char_exc_erg_eV eV. Regenerate the table or " *
                "update PlasmaConstants.char_exc_erg_eV so they match."
        )
    end
    return nothing
end

"""
    H2_Ion_RRCs{FT<:AbstractFloat} <: AbstractSpeciesRRCs{FT}

Container for H2+ ion-related reaction rate coefficient models.
Stores various reaction models for H2+ interactions with background gas.

# Fields
- `Elastic`: Rate coefficient for elastic collisions
- `Charge_Exchange`: Rate coefficient for charge exchange processes
- `Target_Ionization`: Rate coefficient for ionization of target particles
- `Projectile_Dissociation`: Rate coefficient for dissociation of projectile ions
- `Particle_Exchange`: Rate coefficient for particle exchange processes
"""
struct H2_Ion_RRCs{FT <: AbstractFloat} <: AbstractSpeciesRRCs{FT}
    Elastic::RRC_T_ud{FT}
    Charge_Exchange::RRC_T_ud{FT}
    Target_Ionization::RRC_T_ud{FT}
    Projectile_Dissociation::RRC_T_ud{FT}
    Particle_Exchange::RRC_T_ud{FT}

    function H2_Ion_RRCs(iRRCs_T_ud_fileName::String)
        # Create RRC_T_ud objects for each reaction type from the given H5 file
        h5fid = h5open(iRRCs_T_ud_fileName, "r")
        order = T_ud_axis_order(h5fid)
        T_eV = read(h5fid, "T_eV")
        ud_para = read(h5fid, "ud_para")
        srf(name) = read_T_ud_surface(h5fid, name, order)
        Elastic = RRC_T_ud(T_eV, ud_para, srf("Elastic"))
        Charge_Exchange = RRC_T_ud(T_eV, ud_para, srf("Charge_Exchange"))
        Target_Ionization = RRC_T_ud(T_eV, ud_para, srf("Target_Ionization"))
        Projectile_Dissociation = RRC_T_ud(T_eV, ud_para, srf("Projectile_Dissociation"))
        Particle_Exchange = RRC_T_ud(T_eV, ud_para, srf("Particle_Exchange"))
        close(h5fid)

        FT = eltype(T_eV)  # Determine the floating-point type from the data

        return new{FT}(Elastic, Charge_Exchange, Target_Ionization, Projectile_Dissociation, Particle_Exchange)
    end
end

"""
    get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat

Calculate electron reaction rate coefficients for the specified reaction using interpolation.
Automatically selects appropriate physical parameters from the RAPID model.

# Arguments
- `RP::RAPID{FT}`: RAPID plasma model containing physical state variables
- `eRRCs::Electron_RRCs{FT}`: Container of electron reaction rate coefficient models
- `reaction::Symbol`: Symbol specifying which reaction to compute (e.g., :Ionization)

# Returns
- RRC (reaction rate coefficient) values at each spatial point
"""
function get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    return if hasfield(typeof(eRRCs), reaction)
        mass = RP.config.constants.me
        ee = RP.config.constants.ee

        RRC = getfield(eRRCs, reaction)
        if RRC isa RRC_EoverP_Erg
            mean_Ke_eV = @. 1.5 * RP.plasma.Te_eV + 0.5 * mass * RP.plasma.ue_para^2 / ee
            abs_Epara_over_pGas = @. abs(RP.fields.E_para_tot / (RP.plasma.n_H2_gas * RP.plasma.T_gas_eV * ee))
            # No-gas guard: n_H2_gas=0 makes E/p non-finite (NaN if E_para=0 too), and a NaN
            # query returns NaN, poisoning ν_en = n_gas·RRC (0·NaN) → singular Ampère matrix.
            # n_gas scales the rate to 0 anyway, so any finite placeholder is safe.
            @. abs_Epara_over_pGas = ifelse(isfinite(abs_Epara_over_pGas), abs_Epara_over_pGas, zero(FT))
            return RRC.itp((abs_Epara_over_pGas, mean_Ke_eV))
        elseif RRC isa RRC_T_ud
            return RRC.itp((RP.plasma.Te_eV, abs.(RP.plasma.ue_para)))
        end
    else
        throw(ArgumentError("Invalid reaction type: $reaction"))
    end
end

# Convenience dispatch
function get_electron_RRC(RP::RAPID{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    return get_electron_RRC(RP, RP.eRRCs, reaction)
end

"""
    get_H2_ion_RRC(RP::RAPID{FT}, iRRCs::H2_Ion_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat

Calculate H2+ ion reaction rate coefficients for the specified reaction using interpolation.
Automatically selects appropriate physical parameters from the RAPID model.

# Arguments
- `RP::RAPID{FT}`: RAPID plasma model containing physical state variables
- `iRRCs::H2_Ion_RRCs{FT}`: Container of H2+ ion reaction rate coefficient models
- `reaction::Symbol`: Symbol specifying which reaction to compute (e.g., :Elastic)

# Returns
- RRC (reaction rate coefficient) values at each spatial point
"""
function get_H2_ion_RRC(RP::RAPID{FT}, iRRCs::H2_Ion_RRCs{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    return if hasfield(typeof(iRRCs), reaction)
        RRC = getfield(iRRCs, reaction)
        if RRC isa RRC_T_ud
            return RRC.itp((RP.plasma.Ti_eV, abs.(RP.plasma.ui_para)))
        end
    else
        throw(ArgumentError("Invalid reaction type: $reaction"))
    end
end

# Convenience dispatch
function get_H2_ion_RRC(RP::RAPID{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    return get_H2_ion_RRC(RP, RP.iRRCs, reaction)
end

"""
    update_RRCs!(RP::RAPID{FT}) where {FT<:AbstractFloat}

Evaluate the electron reaction rate coefficients on the `(E/p, Ē)` surfaces and store the
corresponding collision frequencies `ν = n_H2_gas · K` on `RP.plasma`.

**This is the only place those tables are queried during a simulation step.** Consumers
read `plasma.ν_en_iz`, `ν_en_mom_tot`, `ν_en_mom_ela`, `ν_en_exc_eff`; they must not call
[`get_electron_RRC`](@ref) themselves. A step that re-queries ends up with the same
physical coefficient evaluated at two different plasma states — the momentum equation
removing drag at one `ν_mom` while the energy equation credits frictional heating at
another — so its discrete energy budget cannot close. See
`internal/docs/src/notes/design/rrc-single-evaluation-point.md`.

Called from the top of [`update_transport_quantities!`](@ref), which runs at the end of a
`run_simulation!` iteration — precisely the state the next `advance_timestep!` enters with.
The frequencies are therefore lagged coefficients at the step-entry state: the standard
semi-implicit choice, and what keeps `u∥` from overshooting at large `dt`.

Frequencies are stored rather than rate coefficients so that a future time-varying
`n_H2_gas` cannot desynchronize consumers within a step. `K` remains recoverable as
`ν / n_H2_gas`.

`Halpha` and the other `(Te, u∥)`-family surfaces are not materialized: they are
diagnostic-only and are still fetched live at snapshot cadence.
"""
function update_RRCs!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    # ∂P/∂Tₑ is assembled from ∂ν/∂Tₑ, and those must be differentiated at the
    # state the frequencies were evaluated at — so they are materialized here and
    # nowhere else, for the same reason the frequencies are. Skipped entirely when
    # no consumer wants them, so the default configuration pays nothing.
    # Only LinearResponse reads these surfaces; under KnownRate nothing does,
    # so nothing pays to build them.
    want_jacobian = RP.flags.scheme.atomic === ExpRB &&
        RP.flags.exprb_eigenvalue === LinearResponse

    if RP.flags.Atomic_Collision
        K_mom_tot = get_electron_RRC(RP, :Total_Momentum)
        K_mom_ela = get_electron_RRC(RP, :Momentum_by_ela)
        K_exc_eff = get_electron_RRC(RP, :Total_Excitation)
        @. pla.ν_en_mom_tot = pla.n_H2_gas * K_mom_tot
        @. pla.ν_en_mom_ela = pla.n_H2_gas * K_mom_ela
        @. pla.ν_en_exc_eff = pla.n_H2_gas * K_exc_eff

        if want_jacobian
            update_rate_jacobian!(RP, :Total_Momentum, pla.dν_dTe.mom_tot)
            update_rate_jacobian!(RP, :Momentum_by_ela, pla.dν_dTe.mom_ela)
            update_rate_jacobian!(RP, :Total_Excitation, pla.dν_dTe.exc_eff)
        end
    end

    # ν_en_iz is consumed by the parallel momentum drag (gated on Atomic_Collision) *and*
    # by the continuity source (gated on src), which are independent flags — so cover the
    # union. With only `src` set, this field previously held a mid-step value left over
    # from the previous iteration, because its unconditional writer sat under
    # Atomic_Collision while its mid-step writer sat under src.
    if RP.flags.Atomic_Collision || RP.flags.src
        if RP.flags.Ionz_method == "Townsend_coeff"
            # Electron avalanche via the Townsend coefficient,
            # α = 3.88 * p * exp(-95 * p / |E_para|)
            α = @. 3.88 * RP.config.prefilled_gas_pressure *
                exp(-95 * RP.config.prefilled_gas_pressure / abs(RP.fields.E_para_tot))
            @. pla.ν_en_iz = α * abs(pla.ue_para)
        elseif RP.flags.Ionz_method == "Xsec"
            K_iz = get_electron_RRC(RP, :Ionization)
            @. pla.ν_en_iz = pla.n_H2_gas * K_iz
            want_jacobian && update_rate_jacobian!(RP, :Ionization, pla.dν_dTe.iz)
        else
            error("Unknown ionization method: $(RP.flags.Ionz_method)")
        end

        # No ionization outside the wall — and therefore no dependence of it on Tₑ
        # there either, or the diagonal would carry a rate the physics does not.
        pla.ν_en_iz[RP.G.nodes.on_out_wall_nids] .= zero(FT)
        want_jacobian && (pla.dν_dTe.iz[RP.G.nodes.on_out_wall_nids] .= zero(FT))
    end

    return RP
end

"""
    update_rate_jacobian!(RP, reaction, out) -> out

Write `∂ν/∂Tₑ = n_H2_gas · (3/2) · ∂K/∂Ē` for one `(E/p, Ē)` surface into `out`.

The `3/2` is `∂Ē/∂Tₑ` for `Ē = 3/2·Tₑ + ½mₑu∥²/e` at fixed `u`; being constant is
what makes this one batched derivative evaluation per surface, at roughly the cost
of the value evaluation the surface already pays.

Queried at the same `(E/p, Ē)` the value path uses — the coordinates are rebuilt
here, including the no-gas guard, only because [`get_electron_RRC`](@ref) returns
values.

**Outside the table in `Ē` the derivative is zero, and is set so explicitly.**
`ClampExtrap` freezes the value there, so `K` genuinely stops depending on `Tₑ`.
The clamp lives here rather than in the interpolant because FastInterpolations
returns the boundary slope for clamped derivative views (upstream bug); `NoExtrap`
on that axis makes a wrong clamp raise instead of reporting a false dependence.
Not a corner case — `apply_electron_density_boundary_conditions!` puts every node
outside the wall at `Ē = 0`.
"""
function update_rate_jacobian!(
        RP::RAPID{FT}, reaction::Symbol, out::AbstractMatrix{FT}
    ) where {FT <: AbstractFloat}
    rrc = getfield(RP.eRRCs, reaction)
    rrc isa RRC_EoverP_Erg ||
        throw(ArgumentError("∂/∂Tₑ is defined for (E/p, Ē) surfaces; $reaction is not one"))

    me, ee = RP.config.constants.me, RP.config.constants.ee
    mean_Ke_eV = @. FT(1.5) * RP.plasma.Te_eV + FT(0.5) * me * RP.plasma.ue_para^2 / ee
    abs_Epara_over_pGas = @. abs(
        RP.fields.E_para_tot / (RP.plasma.n_H2_gas * RP.plasma.T_gas_eV * ee)
    )
    @. abs_Epara_over_pGas = ifelse(
        isfinite(abs_Epara_over_pGas), abs_Epara_over_pGas, zero(FT)
    )

    # Clamp on the Ē axis only. E/p carries no Tₑ dependence, so the interpolant
    # clamps it exactly as the value path does and nothing needs masking there.
    Ē_lo, Ē_hi = first(rrc.Erg_eV), last(rrc.Erg_eV)
    Ē_query = clamp.(mean_Ke_eV, Ē_lo, Ē_hi)

    rrc.dK_dĒ(out, (abs_Epara_over_pGas, Ē_query))
    # Exact equality is the in-range test: `clamp` returns its argument untouched
    # inside the interval and a bound outside it.
    @. out = ifelse(
        mean_Ke_eV == Ē_query, out * RP.plasma.n_H2_gas * FT(1.5), zero(FT)
    )
    return out
end

# Export types and functions for reaction rate coefficients
export update_RRCs!, update_rate_jacobian!, ion_rate_jacobian
export AbstractReactionRateCoefficient
export RRC_EoverP_Erg, RRC_T_ud, RRC_T_ud_gFac
export Electron_RRCs, H2_Ion_RRCs
export get_electron_RRC, get_H2_ion_RRC
