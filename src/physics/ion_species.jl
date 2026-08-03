# What an ion species is, and whether species share a transport equation.
#
# Declarations only — `types.jl` needs them to type a flag and a state field, so
# they are included before it. The machinery that consumes them lives in
# `physics/ion_transport.jl`.
#
# Ion species exchange momentum with each other far faster than they are
# transported: at ne = 1e15 m⁻³, Ti = 1 eV, τ_ii = 1.51 ms against
# τ_transport ≈ 0.515 s, a ratio of 342. Friction damps any relative drift 342×
# faster than transport can build one, so ion species do not diffuse
# independently — they drag each other, which is the same physics as impurity
# pinch and screening. On that reading one effective equation is not a shortcut
# but the correct limit.
#
# Against that: D∥ carries 73 % of the ion flux that reaches the wall at those
# conditions (convection 11 %, D⊥ 15 %), and D∥ is the one mechanism where the
# species genuinely differ. So the mixing error lands on the dominant term.
#
# Both readings are therefore available, and which one runs is a matter of TYPE.
# No call site below `ion_transport_groups` asks which policy is in force.

"""
    IonSpecies{FT}

One ion species: what its transport channels need to know about it.

`mass` and `charge` vary independently across the periodic table — C⁶⁺ and H₂⁺
differ 6× in both, but O⁸⁺ and H₂⁺ differ 8× in charge and 8× in mass while C⁴⁺
and He²⁺ share neither ratio. That independence is why `Zeff` alone cannot stand
in for a species: one scalar cannot carry two independent numbers.
"""
struct IonSpecies{FT <: AbstractFloat}
    name::Symbol
    mass::FT
    charge::Int

    function IonSpecies{FT}(name::Symbol, mass::FT, charge::Int) where {FT <: AbstractFloat}
        mass > zero(FT) || throw(ArgumentError("$name: mass must be positive, got $mass"))
        charge > 0 || throw(ArgumentError("$name: an ion carries charge ≥ 1, got $charge"))
        return new{FT}(name, mass, charge)
    end
end

IonSpecies(name::Symbol, mass::Real, charge::Integer) =
    IonSpecies{float(typeof(mass))}(name, float(mass), Int(charge))

"""
    IonTransportPolicy

Whether ion species share a transport operator, and which one.

The choice is a type rather than a flag value so that it resolves at
`ion_transport_groups` and nowhere else. Everything downstream — assembly,
factorization, the solve — is written once against the groups and never learns
which policy produced them.
"""
abstract type IonTransportPolicy end

"""
    SharedEffectiveTransport()

One operator for every ion species, built from density-weighted effective
channels. **The default.**

Justified by `τ_ii ≪ τ_transport`: friction couples the species into a single
fluid long before transport can separate them. Costs one factorization for any
number of species, with the species entering as extra right-hand sides.

Mixing happens **mechanism by mechanism**, not on the assembled tensor. That
keeps each mechanism's wall ceiling separable — ceilings *add* across mechanisms
while densities *average* across species, and collapsing to a tensor first would
lose the distinction.
"""
struct SharedEffectiveTransport <: IonTransportPolicy end

"""
    PerSpeciesTransport()

Each ion species gets its own operator, from its own channels, with nothing
averaged.

The reference against which [`SharedEffectiveTransport`](@ref) is measured, and
the fallback if the mixing error ever turns out to matter. Costs one
factorization per species.

Measured against the shared policy on a 25×25 box wall (H₂⁺ against a 6× heavier
carbon ion, 20 steps onto a fully absorbing wall, fraction of inventory
surviving): the two differ by ±10 % at a 50:50 mix, and at a realistic 99:1 mix
the bulk is untouched (+0.2 %) while the whole error lands on the trace species
(−18.5 %) — which is what friction coupling says should physically happen to a
trace impurity in a hydrogen background.
"""
struct PerSpeciesTransport <: IonTransportPolicy end
