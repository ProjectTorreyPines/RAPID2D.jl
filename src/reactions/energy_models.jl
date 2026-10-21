# What a reaction does to the electrons' energy, as the reaction catalog declares it: per event
# with its count, or per unit time with the densities.

"""
    ElectronEnergyModel

The electron energy a reaction takes, in one of two forms the electron energy ledger books
differently:

- per event ([`EventEnergy`](@ref), [`CapturedElectron`](@ref)) [J]: multiplied by the event
  count its producer formed, at that count's epoch;
- per unit time ([`ContinuousPower`](@ref)) [W·m³]: multiplied by nₑ and the target density at
  the producer's density epoch, the coefficient at the rate's evaluation point.
"""
abstract type ElectronEnergyModel end

"""
    EventEnergy(electron_loss)

The electrons lose `electron_loss` [J] per event: a threshold or a potential, fixed. The part
that is potential energy (an ionization) the ledger books as chemical potential, not as loss.
"""
struct EventEnergy{FT <: AbstractFloat} <: ElectronEnergyModel
    electron_loss::FT
end

"""
    CapturedElectron()

The electron is captured (a recombination) and takes its kinetic energy out of the electron
fluid: for Maxwellian electrons at Tₑ on a target at rest,

    ⟨σvE⟩/⟨σv⟩ = e Tₑ (3/2 + d ln K/d ln Tₑ)   [J per event]

from the reaction's own rate K (AMJUEL §1.3.8, the H.8–H.2 relation), so it needs a rate with a
derivative. Less than (3/2)eTₑ when slow electrons are captured preferentially.
"""
struct CapturedElectron <: ElectronEnergyModel end

"""
    ContinuousPower(model)

The electrons lose P = L nₑ n_target [W/m³], L [W·m³] an energy-weighted rate model in the
reaction's context: excitation and radiation that no event count carries.
"""
struct ContinuousPower{M <: AbstractRateModel} <: ElectronEnergyModel
    model::M
end

"""
    event_energy!(out, energy, rate, ctx, ee) -> out

The electron energy per event [J] at every node of `ctx`, for a reaction of rate model `rate`
whose electrons take `energy` per event ([`EventEnergy`](@ref), [`CapturedElectron`](@ref)).
"""
event_energy!(out::AbstractArray, energy::EventEnergy, ::AbstractRateModel, ::AbstractRateContext, _) =
    fill!(out, energy.electron_loss)

function event_energy!(out::AbstractArray{FT}, ::CapturedElectron, rate::AbstractRateModel, ctx::TemperatureContext, ee) where {FT}
    derivative_support(rate) isa HasDerivative ||
        throw(ArgumentError("a captured electron's energy needs d ln K/d ln Tₑ, and $(nameof(typeof(rate))) has no derivative"))
    @with_pool pool begin
        K = rate_coefficient!(acquire!(pool, FT, size(out)...), rate, ctx)
        rate_derivative!(out, rate, ctx)
        # e T (3/2 + T K′/K); where K vanishes no electron is captured, and the limit is (3/2)eT
        @. out = ee * ctx.T_eV * (FT(1.5) + ifelse(K > zero(FT), ctx.T_eV * out / K, zero(FT)))
    end
    return out
end
