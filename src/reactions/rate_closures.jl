export RateClosure, KineticClosure, MaxwellianClosure

# How a reaction's rate is averaged over the distribution of the colliding particles: the choice
# `config.closures` holds. Included before `types.jl`.

"""
    RateClosure

How a rate coefficient is averaged over the distribution of the colliding particles. A process
provides its rate in a closure by a method of `rate_data`; a run takes, for each process, the
first closure in `config.closures` the process has data in.
"""
abstract type RateClosure end

"""
    KineticClosure()

Averaged over BREAK's kinetic, non-Maxwellian electron distribution and tabulated in (E/p, Ē):
electron impact on H₂ so far.
"""
struct KineticClosure <: RateClosure end

"""
    MaxwellianClosure()

The traditional closure: averaged over a Maxwellian, or a drifting Maxwellian, at the colliding
species' temperature (fits and (T, u_d) tables).
"""
struct MaxwellianClosure <: RateClosure end
