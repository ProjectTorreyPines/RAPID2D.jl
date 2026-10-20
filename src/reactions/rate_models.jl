export ConstantRate, ArrheniusRate, LogPolynomialRate

# Rate coefficients by model: a reaction's K = ⟨σv⟩ from a table or a fit, evaluated at the
# coordinates of the colliding pair (its context).

# ── where a rate is evaluated ───────────────────────────────────────────────────────────

"""
    AbstractRateContext

The coordinates a rate coefficient is evaluated at, node by node, as the colliding pair defines
them. Each context has an energy coordinate, the one rates are differentiated in: Ē in an
[`ElectronImpactContext`](@ref), T in an [`ElectronDriftContext`](@ref) or a
[`HeavyPairContext`](@ref).
"""
abstract type AbstractRateContext end

"""
    ElectronImpactContext(EoverP, Erg_eV)
    ElectronImpactContext(RP)

Electron impact on the gas, on the (E/p, Ē) surfaces: E/p = |E∥|/(n_gas T_gas e) and the mean
electron energy Ē = (3/2)Tₑ + ½mₑu∥²/e [eV], as [`_eRRC_query_point`](@ref) forms them.
"""
struct ElectronImpactContext{EoverPArray <: AbstractArray, ErgArray <: AbstractArray} <: AbstractRateContext
    EoverP::EoverPArray
    Erg_eV::ErgArray
end

ElectronImpactContext(RP::RAPID) = ElectronImpactContext(_eRRC_query_point(RP)...)

"""
    ElectronDriftContext(T_eV, u)
    ElectronDriftContext(RP)

Electrons by temperature Tₑ [eV] and drift |uₑ∥| [m/s], the axes of the electron (T, u_d)
surfaces.
"""
struct ElectronDriftContext{TArray <: AbstractArray, UArray <: AbstractArray} <: AbstractRateContext
    T_eV::TArray
    u::UArray
end

ElectronDriftContext(RP::RAPID) = ElectronDriftContext(RP.plasma.Te_eV, abs.(RP.plasma.ue_para))

"""
    HeavyPairContext(T_eV, u)
    HeavyPairContext(RP)

An ion on the gas, by the ion's temperature Tᵢ [eV] and drift |uᵢ∥| [m/s], the axes of the ion
(T, u_d) surfaces.
"""
struct HeavyPairContext{TArray <: AbstractArray, UArray <: AbstractArray} <: AbstractRateContext
    T_eV::TArray
    u::UArray
end

HeavyPairContext(RP::RAPID) = HeavyPairContext(RP.plasma.Ti_eV, abs.(RP.plasma.ui_para))

const TemperatureContext = Union{ElectronDriftContext, HeavyPairContext}

"The coordinate a context's rates are differentiated in, node by node."
energy_coordinate(ctx::ElectronImpactContext) = ctx.Erg_eV
energy_coordinate(ctx::TemperatureContext) = ctx.T_eV

# ── models ──────────────────────────────────────────────────────────────────────────────

"""
    AbstractRateModel

A reaction's rate coefficient K [m³/s] at the coordinates of an [`AbstractRateContext`](@ref): a
table ([`TabulatedRate`](@ref)) or a fit ([`ConstantRate`](@ref), [`ArrheniusRate`](@ref),
[`LogPolynomialRate`](@ref)). [`rate_coefficient`](@ref) evaluates it and
[`rate_derivative!`](@ref) differentiates it where [`derivative_support`](@ref) allows.
"""
abstract type AbstractRateModel end

"K, constant [m³/s], in any context."
struct ConstantRate{FT <: AbstractFloat} <: AbstractRateModel
    K::FT
end

"""
    ArrheniusRate(prefactor, T_exponent, E_th, T_range)

K = prefactor · T^T_exponent · exp(−E_th/T) [m³/s], with T and E_th [eV], T the context's
temperature clamped to `T_range`, the fit's validity: outside it K is frozen and its
derivative zero.
"""
struct ArrheniusRate{FT <: AbstractFloat} <: AbstractRateModel
    prefactor::FT
    T_exponent::FT
    E_th::FT
    T_range::NTuple{2, FT}

    function ArrheniusRate(prefactor::FT, T_exponent::FT, E_th::FT, T_range::NTuple{2, FT}) where {FT <: AbstractFloat}
        check_fit_range(T_range)
        return new{FT}(prefactor, T_exponent, E_th, T_range)
    end
end

"""
    LogPolynomialRate(coefficients, T_range)

ln K = Σₖ coefficients[k] (ln T)ᵏ⁻¹ with K [m³/s] and T [eV] the context's temperature clamped
to `T_range`, the fit's validity: outside it K is frozen and its derivative zero. The form of
the Janev–Reiter fits (fitted in cm³/s: add ln 10⁻⁶ to the first coefficient).
"""
struct LogPolynomialRate{FT <: AbstractFloat, N} <: AbstractRateModel
    coefficients::NTuple{N, FT}
    T_range::NTuple{2, FT}

    function LogPolynomialRate(coefficients::NTuple{N, FT}, T_range::NTuple{2, FT}) where {FT <: AbstractFloat, N}
        N >= 1 || throw(ArgumentError("a log-polynomial fit needs at least one coefficient"))
        check_fit_range(T_range)
        return new{FT, N}(coefficients, T_range)
    end
end

function check_fit_range(T_range)
    0 < T_range[1] < T_range[2] || throw(ArgumentError("a fit's range must be 0 < T_lo < T_hi, got $T_range"))
    return nothing
end

"""
    JanevFit(coefficients, T_range, source)

A Janev–Reiter Maxwellian rate fit as the EIRENE databases print it (HYDHEL and AMJUEL, section
H.2): ln⟨σv⟩[cm³/s] = Σₙ₌₀⁸ bₙ (ln T[eV])ⁿ, `coefficients` = (b₀, …, b₈), used on `T_range` [eV].
`LogPolynomialRate(FT, fit)` is its rate in m³/s.
"""
struct JanevFit
    coefficients::NTuple{9, Float64}
    T_range::NTuple{2, Float64}
    source::String
end

"""
    LogPolynomialRate(FT, fit::JanevFit; fraction = 1)

The fit's rate in m³/s (b₀ + ln 10⁻⁶) times `fraction`, a branch of the fitted reaction.
"""
function LogPolynomialRate(::Type{FT}, fit::JanevFit; fraction::Real = 1) where {FT <: AbstractFloat}
    b = fit.coefficients
    return LogPolynomialRate(FT.((b[1] + log(1.0e-6) + log(fraction), Base.tail(b)...)), FT.(fit.T_range))
end

"""
    TabulatedRate(table)

A rate table: an [`RRC_EoverP_Erg`](@ref) in an [`ElectronImpactContext`](@ref), an
[`RRC_T_ud`](@ref) in an [`ElectronDriftContext`](@ref) or a [`HeavyPairContext`](@ref). Any
other pairing has no method.
"""
struct TabulatedRate{R <: AbstractReactionRateCoefficient} <: AbstractRateModel
    table::R
end

"""
    TemperatureScaled(model, factor)

`model` at the context's temperature times `factor`, K(T) = K_model(factor·T): a heavy pair's
rate carried to other masses at equal velocity, `factor = μ_ref/μ`.
"""
struct TemperatureScaled{M <: AbstractRateModel, FT <: AbstractFloat} <: AbstractRateModel
    model::M
    factor::FT
end

abstract type DerivativeSupport end
struct NoDerivative <: DerivativeSupport end
struct HasDerivative <: DerivativeSupport end

"""
    derivative_support(model) -> NoDerivative() | HasDerivative()

Whether `model` has a derivative in its context's energy coordinate ([`rate_derivative!`](@ref)).
"""
derivative_support(::AbstractRateModel) = NoDerivative()
derivative_support(::Union{ConstantRate, ArrheniusRate, LogPolynomialRate}) = HasDerivative()
derivative_support(::TabulatedRate{<:Union{RRC_EoverP_Erg, RRC_T_ud}}) = HasDerivative()
derivative_support(m::TemperatureScaled) = derivative_support(m.model)

# ── evaluation ──────────────────────────────────────────────────────────────────────────

"""
    rate_coefficient(model, ctx) -> Array
    rate_coefficient!(out, model, ctx) -> out

`model`'s K at every node of `ctx` [m³/s], in a new array shaped like the context's coordinates
or in `out`.
"""
rate_coefficient(model::AbstractRateModel, ctx::AbstractRateContext) =
    rate_coefficient!(similar(energy_coordinate(ctx)), model, ctx)

"""
    rate_derivative!(out, model, ctx) -> out

∂K/∂ε at every node of `ctx` into `out`, ε the context's energy coordinate: Ē [eV] in an
[`ElectronImpactContext`](@ref), T [eV] otherwise. A fit is frozen outside its range, so its
derivative is zero there. A table is differentiated at the query as given; outside its energy
axis its value is clamped, so the caller clamps the coordinate to the axis and zeroes the nodes
it moved ([`update_rate_jacobian!`](@ref), [`ion_rate_jacobian`](@ref)).
"""
function rate_derivative! end

rate_coefficient!(out::AbstractArray, m::ConstantRate, ::AbstractRateContext) = fill!(out, m.K)
rate_derivative!(out::AbstractArray, ::ConstantRate, ::AbstractRateContext) = fill!(out, zero(eltype(out)))

function rate_coefficient!(out::AbstractArray, m::ArrheniusRate, ctx::TemperatureContext)
    out .= arrhenius_rate.(ctx.T_eV, m.prefactor, m.T_exponent, m.E_th, m.T_range...)
    return out
end

function rate_derivative!(out::AbstractArray, m::ArrheniusRate, ctx::TemperatureContext)
    out .= arrhenius_slope.(ctx.T_eV, m.prefactor, m.T_exponent, m.E_th, m.T_range...)
    return out
end

function rate_coefficient!(out::AbstractArray, m::LogPolynomialRate, ctx::TemperatureContext)
    out .= log_polynomial_rate.(ctx.T_eV, Ref(m.coefficients), m.T_range...)
    return out
end

function rate_derivative!(out::AbstractArray, m::LogPolynomialRate, ctx::TemperatureContext)
    slope_coefficients = log_derivative_coefficients(m.coefficients)
    out .= log_polynomial_slope.(ctx.T_eV, Ref(m.coefficients), Ref(slope_coefficients), m.T_range...)
    return out
end

@inline function arrhenius_rate(T, prefactor, T_exponent, E_th, T_lo, T_hi)
    Tc = clamp(T, T_lo, T_hi)
    return prefactor * Tc^T_exponent * exp(-E_th / Tc)
end

# ∂K/∂T = K (T_exponent + E_th/T)/T in the range; exact equality is the in-range test, as clamp
# returns its argument untouched inside
@inline function arrhenius_slope(T, prefactor, T_exponent, E_th, T_lo, T_hi)
    Tc = clamp(T, T_lo, T_hi)
    Tc == T || return zero(Tc)
    return prefactor * Tc^T_exponent * exp(-E_th / Tc) * (T_exponent + E_th / Tc) / Tc
end

@inline log_polynomial_rate(T, coefficients, T_lo, T_hi) = exp(evalpoly(log(clamp(T, T_lo, T_hi)), coefficients))

# ∂K/∂T = (K/T) d(ln K)/d(ln T) in the range
@inline function log_polynomial_slope(T, coefficients, slope_coefficients, T_lo, T_hi)
    Tc = clamp(T, T_lo, T_hi)
    Tc == T || return zero(Tc)
    lnT = log(Tc)
    return exp(evalpoly(lnT, coefficients)) * evalpoly(lnT, slope_coefficients) / Tc
end

# The coefficients of d(ln K)/d(ln T) as a polynomial in ln T; a constant for a one-term fit
log_derivative_coefficients(::NTuple{1, FT}) where {FT} = (zero(FT),)
log_derivative_coefficients(coefficients::NTuple{N, FT}) where {N, FT} =
    ntuple(k -> k * coefficients[k + 1], Val(N - 1))

# A table answers on its own interpolants: the allocating forms allocate the output and fill it
# in place, so both forms give the same numbers.
rate_coefficient(m::TabulatedRate{<:RRC_EoverP_Erg}, ctx::ElectronImpactContext) =
    m.table.itp((ctx.EoverP, ctx.Erg_eV))
rate_coefficient!(out::AbstractArray, m::TabulatedRate{<:RRC_EoverP_Erg}, ctx::ElectronImpactContext) =
    m.table.itp(out, (ctx.EoverP, ctx.Erg_eV))
rate_derivative!(out::AbstractArray, m::TabulatedRate{<:RRC_EoverP_Erg}, ctx::ElectronImpactContext) =
    m.table.dK_dĒ(out, (ctx.EoverP, ctx.Erg_eV))

rate_coefficient(m::TabulatedRate{<:RRC_T_ud}, ctx::TemperatureContext) = m.table.itp((ctx.T_eV, ctx.u))
rate_coefficient!(out::AbstractArray, m::TabulatedRate{<:RRC_T_ud}, ctx::TemperatureContext) =
    m.table.itp(out, (ctx.T_eV, ctx.u))
rate_derivative!(out::AbstractArray, m::TabulatedRate{<:RRC_T_ud}, ctx::TemperatureContext) =
    m.table.dK_dT(out, (ctx.T_eV, ctx.u))

function rate_coefficient!(out::AbstractArray, m::TemperatureScaled, ctx::TemperatureContext)
    @with_pool pool begin
        T = acquire!(pool, eltype(ctx.T_eV), size(ctx.T_eV)...)
        T .= m.factor .* ctx.T_eV
        rate_coefficient!(out, m.model, with_temperature(ctx, T))
    end
    return out
end

# d/dT K_model(factor·T) = factor·K′_model(factor·T)
function rate_derivative!(out::AbstractArray, m::TemperatureScaled, ctx::TemperatureContext)
    @with_pool pool begin
        T = acquire!(pool, eltype(ctx.T_eV), size(ctx.T_eV)...)
        T .= m.factor .* ctx.T_eV
        rate_derivative!(out, m.model, with_temperature(ctx, T))
    end
    out .*= m.factor
    return out
end

with_temperature(ctx::ElectronDriftContext, T) = ElectronDriftContext(T, ctx.u)
with_temperature(ctx::HeavyPairContext, T) = HeavyPairContext(T, ctx.u)

# ∂K of the surface `reaction` of a container of tables at `ctx`: a barrier, so the call into
# the surface is specialized however abstractly the caller holds the container
surface_derivative!(out::AbstractArray, rrcs::AbstractSpeciesRRCs, reaction::Symbol, ctx::AbstractRateContext) =
    rate_derivative!(out, TabulatedRate(getfield(rrcs, reaction)), ctx)
