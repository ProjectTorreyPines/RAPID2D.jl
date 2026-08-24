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
    t_axis_bounds(RP, reaction) -> (T_lo, T_hi)

The endpoints of one `(T, u_d)` ion surface's temperature axis [eV], as a concrete
`Tuple{FT,FT}`. The ion-side twin of [`erg_axis_bounds`](@ref), and a function barrier
for the same reason: `RAPID.iRRCs` is the abstract `AbstractSpeciesRRCs{FT}` and
`reaction` selects the field at runtime, so `first(rrc.T_eV)` arrives `::AbstractFloat`
and would enter `clamp.` over the whole grid as a non-concrete scalar operand.

**Separate from [`ion_rate_jacobian`](@ref) on purpose.** That function also asserts
`::Matrix{FT}` on its `dK_dT` call, and that assertion alone makes its RETURN type
concrete — so a test on the return type cannot see whether these scalars are pinned.
Mutation-checked: removing the `::FT` here leaves `ion_rate_jacobian` inferring
`Matrix{Float64}` regardless. This accessor is what makes the scalar layer testable.
"""
@inline function t_axis_bounds(
        RP::RAPID{FT}, reaction::Symbol
    ) where {FT <: AbstractFloat}
    rrc = getfield(RP.iRRCs, reaction)
    rrc isa RRC_T_ud ||
        throw(ArgumentError("the T axis is defined for (T, u_d) surfaces; $reaction is not one"))
    return (first(rrc.T_eV)::FT, last(rrc.T_eV)::FT)
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

    # Two concretising assertions, and BOTH are needed — dropping either leaves this
    # function inferring `Any`, which then propagates into the two whole-grid
    # broadcasts `update_ion_power_jacobian!` builds from its result.
    #   `::FT`        — `RP.iRRCs` is the abstract `AbstractSpeciesRRCs{FT}` and
    #                   `reaction` selects the field at runtime, so `first(rrc.T_eV)`
    #                   arrives `::AbstractFloat` and would enter `clamp.` as a
    #                   non-concrete scalar operand.
    #   `::Matrix{FT}` — `RRC_T_ud` declares `itp` and `dK_dT` with no type at all
    #                   (see the struct), so the call is `Any` however well the
    #                   scalars are pinned.
    # Pinned by `rrc_type_stability_test.jl`; the same barrier idiom as
    # [`erg_axis_bounds`](@ref) and [`mean_energy_floor`](@ref).
    T_lo, T_hi = t_axis_bounds(RP, reaction)
    T_query = clamp.(RP.plasma.Ti_eV, T_lo, T_hi)
    out = rrc.dK_dT((T_query, abs.(RP.plasma.ui_para)))::Matrix{FT}
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

Three ledgers, three weightings of the same `⟨σv⟩` average. See
`internal/docs/src/notes/design/bd-2026-08-governing-equations.md` §2 — read it before
substituting one for another; none of them are interchangeable.

## Particle ledger [m³/s] — each collision counts 1
- `K_iz`: `e + H₂ → 2e + H₂⁺`
- `K_diss_iz`: `e + H₂ → 2e + H⁺ + H⁰`, zero below 35 eV impact
- `K_exc`: EXC-group event rate (singlets + vib + rot)
- `K_diss_exc`: `e + H₂ → 2H⁰ + e`, triplets plus the B/C singlet branching yields

## Momentum ledger [m³/s] — each collision counts `w_mom`, `v_z`-weighted moment
- `K_mom`: drift friction `ν_mom/n_gas` = `⟨σ_mom|v|v_z⟩/⟨v_z⟩`. The ONLY momentum input
  to the solver. Not the density-weighted collision frequency
- `K_mom_by_*`: per-group shares, `K_mom = Σ K_mom_by_*` exactly. **Diagnostics only** —
  they are the sole auditor of `K_mom`, which has no other check

## Energy ledger [W·m³] — each collision counts `Δε`
- `Kerg_ela`: elastic recoil. **`2mₑ/M` and `e` are already inside.** Cold-target: it does
  not vanish at `Tₑ = T_gas`, hence the `(1 − 3T_gas/2Ē)` factor at the use site
- `Kerg_exc`: EXC group. Carries the vib/rot cooling nothing else has. Also consumed
  with the `(1 − 3T_gas/2Ē)` factor at the use site — a deliberate deviation from this
  dataset's own `consume_as` (which specifies no factor), because it too is a one-way,
  ground-state coefficient with no superelastic return. See the code site
  (`update_electron_heating_powers!`) and `RRC_data/README.md` for the full reasoning
- `Kerg_diss_exc`: DISS group, triplet energy only
- `Kerg_tot`: `Kerg_ela + Kerg_exc + Kerg_diss_exc + e(15.426·K_iz + 35.0·K_diss_iz)`,
  assembled by BD from the exported parts, so it closes by construction

Never form `Kerg_x/K_x` at runtime: the DISS pair is deliberately asymmetric and both are
0/0 over most of the grid.

# Other fields
- `Dissoc_Ionz_legacy`: (T,ud) dissociative-ionization surface from eRRCs_T_ud.h5.
  NOT the same quantity as `K_diss_iz`: different coordinates, different data
  generation, no consumer
- `Halpha`: Rate coefficient for Halpha emission
- `Recomb_H2Ion`: Rate coefficient for H2+ recombination
- `Recomb_H3Ion`: Rate coefficient for H3+ recombination
"""
struct Electron_RRCs{FT <: AbstractFloat} <: AbstractSpeciesRRCs{FT}
    # ── particle ledger [m³/s] — count EVENTS
    K_iz::RRC_EoverP_Erg{FT}
    K_diss_iz::RRC_EoverP_Erg{FT}
    K_exc::RRC_EoverP_Erg{FT}
    K_diss_exc::RRC_EoverP_Erg{FT}

    # ── momentum ledger [m³/s] — v_z-weighted drift friction, K_mom = Σ K_mom_by_*
    K_mom::RRC_EoverP_Erg{FT}
    K_mom_by_ela::RRC_EoverP_Erg{FT}
    K_mom_by_exc::RRC_EoverP_Erg{FT}
    K_mom_by_diss_exc::RRC_EoverP_Erg{FT}
    K_mom_by_iz::RRC_EoverP_Erg{FT}
    K_mom_by_diss_iz::RRC_EoverP_Erg{FT}

    # ── energy ledger [W·m³] — carry ENERGY; BD spells these `L_*`
    Kerg_ela::RRC_EoverP_Erg{FT}
    Kerg_exc::RRC_EoverP_Erg{FT}
    Kerg_diss_exc::RRC_EoverP_Erg{FT}
    Kerg_tot::RRC_EoverP_Erg{FT}

    # From eRRCs_T_ud.h5, and NOT the same quantity as `K_diss_iz`: different
    # coordinates, different data generation, no consumer. Named `_legacy` so the
    # collision with the live channel cannot be made by accident.
    Dissoc_Ionz_legacy::RRC_T_ud{FT}
    Halpha::RRC_T_ud{FT}
    Recomb_H2Ion::RRC_T_ud{FT}
    Recomb_H3Ion::RRC_T_ud{FT}

    function Electron_RRCs(eRRC_EoverP_Erg_fileName::String, eRRC_T_ud_fileName::String)
        @assert isfile(eRRC_EoverP_Erg_fileName) "File not found: $eRRC_EoverP_Erg_fileName"
        @assert isfile(eRRC_T_ud_fileName) "File not found: $eRRC_T_ud_fileName"

        # Create RRC_EoverP_Erg objects for each reaction type from the given H5 file
        h5fid = h5open(eRRC_EoverP_Erg_fileName, "r")
        EoverP = read(h5fid, "EoverP")
        Erg_eV = read(h5fid, "Erg_eV")

        # THE BD ↔ RAPID2D NAME MAPPING LIVES HERE AND NOWHERE ELSE.
        # BD's `L_*` are renamed `Kerg_*` because `L_` is already length throughout this
        # codebase (Lc, Lc_tot, L_mixing, L_char, Lpol). Anyone cross-reading BD's docs
        # needs this table; keep it next to the reads.
        #
        #   BD dataset          RAPID2D field        units      manuscript
        #   K_iz                K_iz                 m³/s       K_iz
        #   K_diss_iz           K_diss_iz            m³/s       K_DI
        #   K_exc               K_exc                m³/s       K_exc
        #   K_diss_exc          K_diss_exc           m³/s       K_diss
        #   K_mom, K_mom_by_*   same                 m³/s       K_mom, K_mom,α
        #   L_ela               Kerg_ela             W·m³       K^ε_ela
        #   L_exc               Kerg_exc             W·m³       K^ε_exc
        #   L_diss_exc          Kerg_diss_exc        W·m³       K^ε_diss
        #   L_tot               Kerg_tot             W·m³       K^ε_tot
        srf_eop(name) = RRC_EoverP_Erg(EoverP, Erg_eV, read(h5fid, name))

        K_iz = srf_eop("K_iz")
        K_diss_iz = srf_eop("K_diss_iz")
        K_exc = srf_eop("K_exc")
        K_diss_exc = srf_eop("K_diss_exc")

        K_mom = srf_eop("K_mom")
        K_mom_by_ela = srf_eop("K_mom_by_ela")
        K_mom_by_exc = srf_eop("K_mom_by_exc")
        K_mom_by_diss_exc = srf_eop("K_mom_by_diss_exc")
        K_mom_by_iz = srf_eop("K_mom_by_iz")
        K_mom_by_diss_iz = srf_eop("K_mom_by_diss_iz")

        Kerg_ela = srf_eop("L_ela")
        Kerg_exc = srf_eop("L_exc")
        Kerg_diss_exc = srf_eop("L_diss_exc")
        Kerg_tot = srf_eop("L_tot")
        close(h5fid)

        # Create RRC_T_ud objects for each reaction type from the given H5 file
        h5fid = h5open(eRRC_T_ud_fileName, "r")
        order = T_ud_axis_order(h5fid)
        T_eV = read(h5fid, "T_eV")
        ud_para = read(h5fid, "ud_para")
        srf(name) = read_T_ud_surface(h5fid, name, order)
        Dissoc_Ionz_legacy = RRC_T_ud(T_eV, ud_para, srf("Dissoc_Ionz"))
        Halpha = RRC_T_ud(T_eV, ud_para, srf("Halpha"))
        Recomb_H2Ion = RRC_T_ud(T_eV, ud_para, srf("Recomb_H2Ion"))
        Recomb_H3Ion = RRC_T_ud(T_eV, ud_para, srf("Recomb_H3Ion"))
        close(h5fid)


        FT = eltype(EoverP)  # Determine the floating-point type from the data

        return new{FT}(
            K_iz, K_diss_iz, K_exc, K_diss_exc,
            K_mom, K_mom_by_ela, K_mom_by_exc, K_mom_by_diss_exc,
            K_mom_by_iz, K_mom_by_diss_iz,
            Kerg_ela, Kerg_exc, Kerg_diss_exc, Kerg_tot,
            Dissoc_Ionz_legacy, Halpha, Recomb_H2Ion, Recomb_H3Ion,
        )
    end
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
    _eRRC_query_point(RP) -> (E/p, Ē)

Where every `RRC_EoverP_Erg` surface is evaluated: `Ē = (3/2)Tₑ + ½mₑu∥²/e` and
`E/p = |E∥|/(n_gas·T_gas·e)`.

One definition, because [`update_rate_jacobian!`](@ref) is a derivative of
[`get_electron_RRC`](@ref) only if both ask the table the same question.

Includes the no-gas guard: `n_H2_gas = 0` makes the ratio non-finite (`NaN` when
`E∥ = 0` as well), and a `NaN` query poisons `ν = n_gas·K` through `0·NaN`, landing a
singular row in the Ampère matrix. `n_gas` scales the rate to zero regardless, so any
finite placeholder is safe.
"""
function _eRRC_query_point(RP::RAPID{FT}) where {FT <: AbstractFloat}
    me, ee = RP.config.constants.me, RP.config.constants.ee
    pla = RP.plasma
    mean_Ke_eV = @. FT(1.5) * pla.Te_eV + FT(0.5) * me * pla.ue_para^2 / ee
    abs_Epara_over_pGas = @. abs(
        RP.fields.E_para_tot / (pla.n_H2_gas * pla.T_gas_eV * ee)
    )
    @. abs_Epara_over_pGas = ifelse(
        isfinite(abs_Epara_over_pGas), abs_Epara_over_pGas, zero(FT)
    )
    return (abs_Epara_over_pGas, mean_Ke_eV)
end

"""
    get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where FT<:AbstractFloat

Calculate electron reaction rate coefficients for the specified reaction using interpolation.
Automatically selects appropriate physical parameters from the RAPID model.

# Arguments
- `RP::RAPID{FT}`: RAPID plasma model containing physical state variables
- `eRRCs::Electron_RRCs{FT}`: Container of electron reaction rate coefficient models
- `reaction::Symbol`: Symbol specifying which reaction to compute (e.g., :K_iz)

# Returns
- RRC (reaction rate coefficient) values at each spatial point
"""
function get_electron_RRC(RP::RAPID{FT}, eRRCs::Electron_RRCs{FT}, reaction::Symbol) where {FT <: AbstractFloat}
    return if hasfield(typeof(eRRCs), reaction)
        RRC = getfield(eRRCs, reaction)
        if RRC isa RRC_EoverP_Erg
            return RRC.itp(_eRRC_query_point(RP))
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
read `plasma.ν_en_iz`, `ν_en_diss_iz`, `ν_en_iz_tot`, `ν_en_mom_tot`, `ν_en_mom_ela`, `P_en_ela`,
`P_en_exc`, `P_en_diss_exc`; they must not call
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
    # ∂P/∂Tₑ is assembled from ∂ν/∂Tₑ, which must be differentiated at the state the
    # frequencies were evaluated at — so it is materialized here and nowhere else, for
    # the same reason they are. Only FullLinearResponse reads it, so the default pays
    # nothing.
    want_jacobian = RP.flags.scheme.atomic === ExpRB &&
        RP.flags.exprb_eigenvalue === FullLinearResponse
    # Stamp the policy this step ran under: both flags stay mutable after `initialize!`,
    # so a consumer re-reading them could get one that has moved since.
    pla.dν_dTe.fresh = want_jacobian

    if RP.flags.Atomic_Collision
        K_mom_tot = get_electron_RRC(RP, :K_mom)
        K_mom_ela = get_electron_RRC(RP, :K_mom_by_ela)
        @. pla.ν_en_mom_tot = pla.n_H2_gas * K_mom_tot
        @. pla.ν_en_mom_ela = pla.n_H2_gas * K_mom_ela

        if want_jacobian
            update_rate_jacobian!(RP, :K_mom, pla.dν_dTe.mom_tot)
            update_rate_jacobian!(RP, :K_mom_by_ela, pla.dν_dTe.mom_ela)
        end

        # Energy ledger. `update_rate_jacobian!` already multiplies by n_H2_gas·(3/2),
        # which is ∂/∂Tₑ through Ē for any surface on this grid — so it serves the
        # Kerg_* surfaces unchanged.
        Kerg_ela = get_electron_RRC(RP, :Kerg_ela)
        Kerg_exc = get_electron_RRC(RP, :Kerg_exc)
        Kerg_diss_exc = get_electron_RRC(RP, :Kerg_diss_exc)
        @. pla.P_en_ela = pla.n_H2_gas * Kerg_ela
        @. pla.P_en_exc = pla.n_H2_gas * Kerg_exc
        @. pla.P_en_diss_exc = pla.n_H2_gas * Kerg_diss_exc

        if want_jacobian
            update_rate_jacobian!(RP, :Kerg_ela, pla.dν_dTe.ela_erg)
            update_rate_jacobian!(RP, :Kerg_exc, pla.dν_dTe.exc_erg)
            update_rate_jacobian!(RP, :Kerg_diss_exc, pla.dν_dTe.diss_exc_erg)
        end
    end

    # The UNION, not either flag alone: ν_en_iz feeds both the momentum drag (gated on
    # Atomic_Collision) and the continuity source (gated on src). With only `src` set,
    # this field used to hold a stale mid-step value from the previous iteration.
    if RP.flags.Atomic_Collision || RP.flags.src
        K_iz = get_electron_RRC(RP, :K_iz)
        @. pla.ν_en_iz = pla.n_H2_gas * K_iz
        want_jacobian && update_rate_jacobian!(RP, :K_iz, pla.dν_dTe.iz)

        K_diss_iz = get_electron_RRC(RP, :K_diss_iz)
        @. pla.ν_en_diss_iz = pla.n_H2_gas * K_diss_iz
        want_jacobian && update_rate_jacobian!(RP, :K_diss_iz, pla.dν_dTe.diss_iz)

        # No ionization outside the wall — and therefore no dependence of it on Tₑ
        # there either, or the diagonal would carry a rate the physics does not.
        # BOTH channels: H⁺ production must die at the same boundary H₂⁺ production does.
        pla.ν_en_iz[RP.G.nodes.on_out_wall_nids] .= zero(FT)
        pla.ν_en_diss_iz[RP.G.nodes.on_out_wall_nids] .= zero(FT)
        if want_jacobian
            pla.dν_dTe.iz[RP.G.nodes.on_out_wall_nids] .= zero(FT)
            pla.dν_dTe.diss_iz[RP.G.nodes.on_out_wall_nids] .= zero(FT)
        end

        # ELECTRON production total: both channels make exactly one electron per event.
        # Continuity, dilution and the growth exponent take this. Under the INTERIM(diz-ion-species)
        # (REACTION_STOICHIOMETRY.diz, until H⁺ is a transportable species) the H₂⁺ ion
        # sources take it too, because DI's ion is booked to H₂⁺ as well — see the
        # field's docstring in types.jl and the comment on REACTION_STOICHIOMETRY.diz.
        @. pla.ν_en_iz_tot = pla.ν_en_iz + pla.ν_en_diss_iz
    end

    return RP
end

"""
    erg_axis_bounds(RP, reaction) -> (Ē_lo, Ē_hi)

The endpoints of one `(E/p, Ē)` surface's `Ē` axis [eV], as a concrete `Tuple{FT,FT}`.

**A function barrier, not a convenience** — the same one [`mean_energy_floor`](@ref) is,
and for the same reason twice over. `RAPID.eRRCs` is declared as the abstract
`AbstractSpeciesRRCs{FT}`, and `reaction` selects the field at RUNTIME, so
`getfield(RP.eRRCs, reaction).Erg_eV` infers `Vector{FT} where FT<:AbstractFloat` and
`first` of it comes back `::AbstractFloat`. Handing that to
`clamp.(mean_Ke_eV, Ē_lo, Ē_hi)` costs the fused kernel its specialization, on every
node, on every call. The `::FT` is what makes it concrete again.

**The `isa` test below does not do that job.** `rrc isa RRC_EoverP_Erg` narrows to the
UnionAll, not to `RRC_EoverP_Erg{FT}`, so it validates without concretising — which is
exactly the premise a review once cleared this site on. The annotations are load-bearing;
the `isa` is only an error message.

Every `RRC_EoverP_Erg` shares the one `Erg_eV` vector the constructor read, so the pair
is the same for every surface and `Ē_lo` is [`mean_energy_floor`](@ref).
"""
@inline function erg_axis_bounds(
        RP::RAPID{FT}, reaction::Symbol
    ) where {FT <: AbstractFloat}
    rrc = getfield(RP.eRRCs, reaction)
    rrc isa RRC_EoverP_Erg ||
        throw(ArgumentError("the Ē axis is defined for (E/p, Ē) surfaces; $reaction is not one"))
    return (first(rrc.Erg_eV)::FT, last(rrc.Erg_eV)::FT)
end

"""
    update_rate_jacobian!(RP, reaction, out) -> out

Write `∂ν/∂Tₑ = n_H2_gas · (3/2) · ∂K/∂Ē` for one `(E/p, Ē)` surface into `out`.

The `3/2` is `∂Ē/∂Tₑ` for `Ē = 3/2·Tₑ + ½mₑu∥²/e` at fixed `u`. Being constant is what
makes this one batched derivative evaluation per surface, at roughly the cost of the
value evaluation already paid. Queried at [`_eRRC_query_point`](@ref) — the same
function the value path calls, so the two cannot drift apart.

**Outside the table in `Ē` the derivative is zero, and is set so explicitly** —
`ClampExtrap` freezes the value there, so `K` genuinely stops depending on `Tₑ`. The
clamp lives here because FastInterpolations returns the boundary slope for clamped
derivative views (upstream bug); `NoExtrap` on that axis makes a wrong clamp raise
instead of reporting a false dependence. Not a corner case:
`apply_electron_density_boundary_conditions!` puts every node outside the wall at
`Ē = 0`.
"""
function update_rate_jacobian!(
        RP::RAPID{FT}, reaction::Symbol, out::AbstractMatrix{FT}
    ) where {FT <: AbstractFloat}
    rrc = getfield(RP.eRRCs, reaction)
    rrc isa RRC_EoverP_Erg ||
        throw(ArgumentError("∂/∂Tₑ is defined for (E/p, Ē) surfaces; $reaction is not one"))

    abs_Epara_over_pGas, mean_Ke_eV = _eRRC_query_point(RP)

    # Clamp on the Ē axis only. E/p carries no Tₑ dependence, so the interpolant
    # clamps it exactly as the value path does and nothing needs masking there.
    # Through `erg_axis_bounds` rather than off `rrc` directly: the bounds are scalar
    # operands of a whole-grid broadcast, and read off the abstract field they arrive
    # `::AbstractFloat`. See that function; pinned by `rrc_type_stability_test.jl`.
    Ē_lo, Ē_hi = erg_axis_bounds(RP, reaction)
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
export update_RRCs!, update_rate_jacobian!, ion_rate_jacobian, erg_axis_bounds, t_axis_bounds
export AbstractReactionRateCoefficient
export RRC_EoverP_Erg, RRC_T_ud, RRC_T_ud_gFac
export Electron_RRCs, H2_Ion_RRCs
export get_electron_RRC, get_H2_ion_RRC
