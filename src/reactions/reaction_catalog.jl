export Reaction, Collision, ElectronSolve, LocalNetwork, RateData, LangevinCapture
export SameForIsotopes, EqualVelocity, ReferenceOnly, ResonantChargeExchange, ElasticScattering
export ReactionProcess, CollisionProcess, RateSources, LadderRates, ladder_rates

# Reactions and collisions among species, and their assembly for a run: a reaction is active
# when the run carries every species it involves, and its rate is resolved for the run's
# hydrogen isotope. Assembled once, at `initialize!`.

# ── who counts a reaction's events ──────────────────────────────────────────────────────

"""
    ReactionProducer

The solve that counts a reaction's events: the electron continuity solve
([`ElectronSolve`](@ref)) or the local reaction network ([`LocalNetwork`](@ref)).
"""
abstract type ReactionProducer end

"""
    ElectronSolve()

Counted by the electron continuity solve at its exposure Δt[(1−θ)nₑⁿ + θnₑⁿ⁺¹]: electron impact
on a neutral, whose electrons are born in the same implicit solve.
"""
struct ElectronSolve <: ReactionProducer end

"""
    LocalNetwork(; implicit, frozen = nothing)

Solved by the local reaction network, implicit in the ion `implicit`. The electron is a catalyst
at its declared value; `frozen`, the neutral second reactant of a heavy pair, is held at its
step-entry density and its consumption taken out after the solve.
"""
struct LocalNetwork{I <: IonSpecies, F <: Union{Nothing, NeutralSpecies}} <: ReactionProducer
    implicit::I
    frozen::F
end

LocalNetwork(; implicit::IonSpecies, frozen::Union{Nothing, NeutralSpecies} = nothing) = LocalNetwork(implicit, frozen)

# ── rates and isotopes ──────────────────────────────────────────────────────────────────

"""
    IsotopeRule

How rate data made for one hydrogen isotope carry to the run's: [`SameForIsotopes`](@ref),
[`EqualVelocity`](@ref) or [`ReferenceOnly`](@ref).
"""
abstract type IsotopeRule end

"Electron impact: the heavy partner's mass does not enter the rate."
struct SameForIsotopes <: IsotopeRule end

"""
    EqualVelocity(reference)

A heavy pair whose cross section is a function of the relative velocity: data made for the
`reference` isotope hold at reduced mass μ as K(T) = K_ref(T μ_ref/μ). Exact for resonant charge
exchange on a target at rest, an approximation for elastic scattering.
"""
struct EqualVelocity{H <: AbstractHydrogenIsotope} <: IsotopeRule
    reference::H
end

"""
    ReferenceOnly(reference)

Data that hold for the `reference` isotope alone: a run of another isotope is refused until
`config.rates` gives the rate.
"""
struct ReferenceOnly{H <: AbstractHydrogenIsotope} <: IsotopeRule
    reference::H
end

"""
    RateData(model, rule, source)

A catalog entry's rate: the model, how it carries to other isotopes ([`IsotopeRule`](@ref)) and
where it comes from (database, section, reaction).
"""
struct RateData{M <: AbstractRateModel, I <: IsotopeRule}
    model::M
    rule::I
    source::String
end

"""
    LangevinCapture(polarizability, source)

The Langevin capture rate of an ion on a neutral of polarizability volume α [m³],
K = 2π √(α e²/(4πε₀ μ)) [m³/s]: independent of temperature, and of the isotope through μ, the
pair's reduced mass.
"""
struct LangevinCapture
    polarizability::Float64
    source::String
end

rate_source(d::Union{RateData, LangevinCapture}) = d.source

# ── reactions ───────────────────────────────────────────────────────────────────────────

"""
    Reaction(label, reactants => products; rate, producer, energy = nothing)

A reaction among species, written with counts: `(electron => 1, H2[1] => 1) =>
(electron => 1, H[1] => 1, H[0] => 1)`. Binary (two reactants, K [m³/s]); `rate` a
[`RateData`](@ref) or [`LangevinCapture`](@ref); `producer` the solve that counts its events
([`ReactionProducer`](@ref)); `energy` what its electrons take ([`ElectronEnergyModel`](@ref)),
`nothing` for a reaction without electrons. Throws unless charge and each element's nuclei are
conserved and the producer fits the reactants.
"""
struct Reaction{RE <: Tuple, PR <: Tuple, R, P <: ReactionProducer, E <: Union{Nothing, ElectronEnergyModel}}
    label::Symbol
    reactants::RE
    products::PR
    rate::R
    producer::P
    energy::E

    function Reaction(label::Symbol, stoichiometry::Pair{<:Tuple, <:Tuple}; rate, producer::ReactionProducer, energy = nothing)
        reactants, products = stoichiometry
        check_stoichiometry(label, reactants, products)
        check_producer(label, producer, reactants, energy)
        return new{typeof(reactants), typeof(products), typeof(rate), typeof(producer), typeof(energy)}(
            label, reactants, products, rate, producer, energy
        )
    end
end

participant_key(sp::AbstractSpecies) = species_key(sp)
participant_key(::Electron) = :electron

nuclei(sp::AbstractSpecies) = composition(sp.parent)
nuclei(::Electron) = ()

net_charge(side) = sum(entry -> last(entry) * charge(first(entry)), side; init = 0)

function nuclei_count(side)
    count = Dict{Symbol, Int}()
    for (x, n) in side, (element, k) in nuclei(x)
        count[element] = get(count, element, 0) + n * k
    end
    return count
end

electron_change(r::Reaction) = electron_count(r.products) - electron_count(r.reactants)
electron_count(side) = sum(entry -> first(entry) isa Electron ? last(entry) : 0, side; init = 0)
has_electron(side) = any(entry -> first(entry) isa Electron, side)   # from the types alone

heavy_participants(side) = map(first, filter(entry -> first(entry) isa AbstractSpecies, side))

function check_stoichiometry(label, reactants, products)
    for side in (reactants, products), entry in side
        (entry isa Pair && first(entry) isa Union{AbstractSpecies, Electron} && last(entry) isa Integer && last(entry) >= 1) ||
            throw(ArgumentError("$label: a stoichiometry entry is `species => count` with count ≥ 1, got $(repr(entry))"))
    end
    for side in (reactants, products)
        allunique(map(entry -> participant_key(first(entry)), side)) ||
            throw(ArgumentError("$label: a species appears twice on one side; give it a count instead"))
    end
    sum(last, reactants) == 2 ||
        throw(ArgumentError("$label: only binary reactions are supported (two reactants, K in m³/s)"))
    net_charge(reactants) == net_charge(products) ||
        throw(ArgumentError("$label: charge is not conserved ($(net_charge(reactants)) → $(net_charge(products)))"))
    nuclei_count(reactants) == nuclei_count(products) ||
        throw(ArgumentError("$label: nuclei are not conserved ($(nuclei_count(reactants)) → $(nuclei_count(products)))"))
    return nothing
end

function check_producer(label, ::ElectronSolve, reactants, energy)
    heavy = heavy_participants(reactants)
    (electron_count(reactants) == 1 && length(heavy) == 1 && only(heavy) isa NeutralSpecies) ||
        throw(ArgumentError("$label: the electron solve counts electron impact on a neutral"))
    energy isa ElectronEnergyModel ||
        throw(ArgumentError("$label: electron impact declares the electrons' energy (`energy = ...`)"))
    return nothing
end

function check_producer(label, p::LocalNetwork, reactants, energy)
    heavy = heavy_participants(reactants)
    p.implicit in heavy || throw(ArgumentError("$label: the implicit species $(species_label(p.implicit)) is not a reactant"))
    if electron_count(reactants) == 1
        isnothing(p.frozen) || throw(ArgumentError("$label: electron impact holds no partner frozen"))
        energy isa ElectronEnergyModel ||
            throw(ArgumentError("$label: electron impact declares the electrons' energy (`energy = ...`)"))
    else
        partner = filter(sp -> sp != p.implicit, heavy)
        (length(partner) == 1 && p.frozen == only(partner)) ||
            throw(ArgumentError("$label: a heavy pair holds its other reactant frozen (`frozen = ...`)"))
        isnothing(energy) || throw(ArgumentError("$label: a reaction without electrons takes no electron energy"))
    end
    return nothing
end

# ── collisions ──────────────────────────────────────────────────────────────────────────

"""
    CollisionMoment

What an ion–neutral collision moves, per collision at rate K: [`ResonantChargeExchange`](@ref)
or [`ElasticScattering`](@ref).
"""
abstract type CollisionMoment end

"""
    ResonantChargeExchange()

The ion and the neutral trade identities: the ion takes the neutral's velocity and temperature,
mᵢ(u_n − uᵢ) of momentum and (3/2)e(T_n − Tᵢ) of energy per collision.
"""
struct ResonantChargeExchange <: CollisionMoment end

"""
    ElasticScattering()

Isotropic scattering in the centre of mass, K from the total cross section: per collision the
ion gives up the fraction μ/mᵢ of its momentum relative to the neutral, ½ for equal masses.
"""
struct ElasticScattering <: CollisionMoment end

"""
    Collision(label, ion, neutral; rate, moment)

An ion–neutral collision: no density changes, momentum and energy move as `moment` says
([`CollisionMoment`](@ref)), at rate K from `rate` ([`RateData`](@ref)).
"""
struct Collision{I <: Ion, N <: Neutral, R, M <: CollisionMoment}
    label::Symbol
    ion::I
    neutral::N
    rate::R
    moment::M
end

Collision(label::Symbol, ion::Ion, neutral::Neutral; rate, moment::CollisionMoment) =
    Collision(label, ion, neutral, rate, moment)

# ── processes: a reaction as physics, its rate by closure ───────────────────────────────

"""
    ReactionProcess

A reaction as physics, apart from where its rate comes from. Its label, stoichiometry and
producer are methods of the process ([`process_label`](@ref), [`stoichiometry`](@ref),
[`producer`](@ref)); its rate and its electrons' energy are methods of the process and a
[`RateClosure`](@ref) ([`rate_data`](@ref), [`electron_energy`](@ref)). A new reaction is a type
with these methods; a new source for an existing rate is a method for another closure, taken
wherever `config.closures` prefers it.
"""
abstract type ReactionProcess end

"""
    CollisionProcess

An ion–neutral collision as physics: [`process_label`](@ref), [`collision_pair`](@ref) and
[`collision_moment`](@ref) are methods of the process, and its rate a method of
[`rate_data`](@ref) for each closure it has data in.
"""
abstract type CollisionProcess end

const Process = Union{ReactionProcess, CollisionProcess}

"""
    RateSources(eRRCs, iRRCs, constants)

What the data methods of a process may read: the run's electron and ion tables and its
constants.
"""
struct RateSources{E, I, FT <: AbstractFloat}
    eRRCs::E
    iRRCs::I
    constants::PlasmaConstants{FT}
end

float_type(::RateSources{<:Any, <:Any, FT}) where {FT} = FT

"A process's label, the name `config.rates` and the report use."
function process_label end

"A reaction process's stoichiometry, `reactants => products` with counts."
function stoichiometry end

"The solve that counts a reaction process's events ([`ReactionProducer`](@ref))."
function producer end

"A collision process's (ion, neutral)."
function collision_pair end

"What a collision process moves ([`CollisionMoment`](@ref))."
function collision_moment end

"""
    rate_data(process, closure, sources) -> RateData | LangevinCapture

`process`'s rate in `closure`, from `sources`. A process defines it for each closure it has data
in; one defined for `::RateClosure` holds in every closure.
"""
function rate_data end

"""
    electron_energy(process, closure, sources) -> ElectronEnergyModel | nothing

What `process`'s electrons lose ([`ElectronEnergyModel`](@ref)), from the same data as its rate
in `closure`; `nothing` for a process without electrons.
"""
function electron_energy end

"Whether `process` has a rate in `closure`."
has_rate(process::Process, closure::RateClosure, sources::RateSources) =
    hasmethod(rate_data, Tuple{typeof(process), typeof(closure), typeof(sources)})

"""
    preferred_closure(process, closures, sources) -> RateClosure

The first of `closures` that `process` has a rate in. Throws if there is none.
"""
function preferred_closure(process::Process, closures::Tuple, sources::RateSources)
    for closure in closures
        has_rate(process, closure, sources) && return closure
    end
    throw(
        ArgumentError(
            "$(process_label(process)) has no rate in any of the closures $(join(map(c -> nameof(typeof(c)), closures), ", "))"
        )
    )
end

"The reaction of `process`, with its rate and electron energy in `closure`."
Reaction(process::ReactionProcess, closure::RateClosure, sources::RateSources) = Reaction(
    process_label(process), stoichiometry(process);
    rate = rate_data(process, closure, sources), producer = producer(process),
    energy = electron_energy(process, closure, sources),
)

"The collision of `process`, with its rate in `closure`."
Collision(process::CollisionProcess, closure::RateClosure, sources::RateSources) = Collision(
    process_label(process), collision_pair(process)...;
    rate = rate_data(process, closure, sources), moment = collision_moment(process),
)

# The heavy species a process involves
function process_species(process::ReactionProcess)
    reactants, products = stoichiometry(process)
    return (heavy_participants(reactants)..., heavy_participants(products)...)
end
process_species(process::CollisionProcess) = collision_pair(process)

# ── the charge-state ladder of an element ───────────────────────────────────────────────

"""
    LadderRates(ionization, recombination, potentials_eV, sources)

An element's charge-state ladder 0…Z, step by step: `ionization[z+1]` the rate model of
e + X[z] → 2e + X[z+1], `recombination[z+1]` that of e + X[z+1] → X[z], `potentials_eV[z+1]`
the ionization potential of X[z] [eV], and `sources` those of the ionization and the
recombination rates. Electron impact, the same for every hydrogen isotope.
"""
struct LadderRates{I <: Tuple, R <: Tuple, P <: Tuple}
    ionization::I
    recombination::R
    potentials_eV::P
    sources::NTuple{2, String}
end

"""
    ladder_rates(element, closure, FT) -> LadderRates

The rates of `element`'s charge-state ladder in `closure`, in floating-point type `FT`. An element
defines it for each closure it has data in.
"""
function ladder_rates end

"""
    LadderIonization(element, z)

e + X[z] → 2e + X[z+1], a step of `element`'s ladder, labelled `iz<X><z>` (`izH0`): counted by the
electron solve from the neutral, by the network above it.
"""
struct LadderIonization{P <: Element} <: ReactionProcess
    element::P
    z::Int
end

"""
    LadderRecombination(element, z)

e + X[z] → X[z−1], a step of `element`'s ladder, labelled `rec<X><z>` (`recH1`), solved by the
network.
"""
struct LadderRecombination{P <: Element} <: ReactionProcess
    element::P
    z::Int
end

const LadderProcess = Union{LadderIonization, LadderRecombination}

process_label(p::LadderIonization) = Symbol(:iz, parent_name(p.element), p.z)
process_label(p::LadderRecombination) = Symbol(:rec, parent_name(p.element), p.z)

stoichiometry(p::LadderIonization) =
    (Particles.electron => 1, p.element[p.z] => 1) => (Particles.electron => 2, p.element[p.z + 1] => 1)
stoichiometry(p::LadderRecombination) = (Particles.electron => 1, p.element[p.z] => 1) => (p.element[p.z - 1] => 1,)

producer(p::LadderIonization) = p.z == 0 ? ElectronSolve() : LocalNetwork(implicit = p.element[p.z])
producer(p::LadderRecombination) = LocalNetwork(implicit = p.element[p.z])

# A ladder has data in the closures its element has ladder rates in
has_rate(p::LadderProcess, closure::RateClosure, sources::RateSources) =
    hasmethod(ladder_rates, Tuple{typeof(p.element), typeof(closure), Type{float_type(sources)}})

function rate_data(p::LadderIonization, closure::RateClosure, sources::RateSources)
    rates = checked_ladder_rates(p.element, closure, sources)
    return RateData(rates.ionization[p.z + 1], SameForIsotopes(), rates.sources[1])
end

function rate_data(p::LadderRecombination, closure::RateClosure, sources::RateSources)
    rates = checked_ladder_rates(p.element, closure, sources)
    return RateData(rates.recombination[p.z], SameForIsotopes(), rates.sources[2])
end

electron_energy(p::LadderIonization, closure::RateClosure, sources::RateSources) =
    EventEnergy(checked_ladder_rates(p.element, closure, sources).potentials_eV[p.z + 1] * sources.constants.ee)
electron_energy(::LadderRecombination, ::RateClosure, ::RateSources) = CapturedElectron()

function checked_ladder_rates(element::Element, closure::RateClosure, sources::RateSources)
    rates = ladder_rates(element, closure, float_type(sources))
    Z = last(allowed_charges(element))
    length(rates.ionization) == length(rates.recombination) == length(rates.potentials_eV) == Z ||
        throw(ArgumentError("$(parent_name(element)): a ladder of $Z steps takes $Z rates of each kind and $Z potentials"))
    return rates
end

"The processes of `element`'s ladder, step by step."
ladder_processes(element::Element) = Tuple(
    Iterators.flatten(
        (LadderIonization(element, z), LadderRecombination(element, z + 1)) for z in 0:(last(allowed_charges(element)) - 1)
    )
)

# The ladder processes of the elements a list carries
function element_ladders(list::Tuple)
    elements = unique(sp.parent for sp in list if sp.parent isa Element)
    return foldl((processes, element) -> (processes..., ladder_processes(element)...), elements; init = ())
end

# ── assembly ────────────────────────────────────────────────────────────────────────────

"""
    ActiveReaction(reaction, rate, source, closure)

A reaction whose species the run all carries, with its rate model for the run's isotope, where
that rate comes from, and the closure its data are in.
"""
struct ActiveReaction{R <: Reaction, M <: AbstractRateModel, CL <: RateClosure}
    reaction::R
    rate::M
    source::String
    closure::CL
end

"A collision whose two species the run carries, with its rate for the run's isotope."
struct ActiveCollision{X <: Collision, M <: AbstractRateModel, CL <: RateClosure}
    collision::X
    rate::M
    source::String
    closure::CL
end

"""
    AssemblyReport

What assembling a run's species left out: each reaction or collision the run lacks a species
of, with those species, and the warnings (a main path of the default hydrogen list switched
off, no reaction that makes electrons).
"""
struct AssemblyReport
    inactive::Vector{Pair{Symbol, Vector{Symbol}}}
    warnings::Vector{String}
end

"""
    assemble_species_system(list, isotope, constants, eRRCs, iRRCs; closures, processes = (), rates = ()) -> SpeciesSystem

The species system of `list`: its registry, and the processes of the catalog (the hydrogen
network, the ladder of every element listed, and `processes`) whose species it all carries.
Each takes its rate from the first of `closures` it has data in (kinetic, then Maxwellian, by
default), resolved for `isotope`, or from `rates` (`label => model` pairs). Throws for a process
with no rate in any of `closures`, an override that names no active process, or data that cannot
be carried to the isotope.
"""
function assemble_species_system(
        list::Tuple, iso::AbstractHydrogenIsotope, constants::PlasmaConstants{FT}, eRRCs, iRRCs;
        closures::Tuple = (KineticClosure(), MaxwellianClosure()), processes::Tuple = (), rates::Tuple = (),
    ) where {FT <: AbstractFloat}
    registry = build_registry(list, iso, constants)
    sources = RateSources(eRRCs, iRRCs, constants)
    all(c -> c isa RateClosure, closures) || throw(ArgumentError("`config.closures` takes rate closures, got $closures"))
    all(p -> p isa Process, processes) ||
        throw(ArgumentError("`config.processes` takes reaction and collision processes, got $processes"))
    catalog = (HYDROGEN_PROCESSES..., element_ladders(list)..., processes...)
    labels = map(process_label, catalog)
    allunique(labels) || throw(ArgumentError("two reactions or collisions share a label in ($(join(labels, ", ")))"))
    overrides = rate_overrides(rates)

    present(p) = isempty(missing_species(registry, p))
    function activate(p::ReactionProcess)
        closure = preferred_closure(p, closures, sources)
        reaction = Reaction(p, closure, sources)
        return ActiveReaction(reaction, active_rate(reaction, overrides, registry, constants)..., closure)
    end
    function activate(p::CollisionProcess)
        closure = preferred_closure(p, closures, sources)
        collision = Collision(p, closure, sources)
        return ActiveCollision(collision, active_rate(collision, overrides, registry, constants)..., closure)
    end
    active = filter(present, catalog)
    active_reactions = map(activate, filter(p -> p isa ReactionProcess, active))
    active_collisions = map(activate, filter(p -> p isa CollisionProcess, active))

    active_labels = Set{Symbol}(map(process_label, active))
    unknown = setdiff(keys(overrides), active_labels)
    isempty(unknown) || throw(
        ArgumentError(
            "`config.rates` gives $(join(unknown, ", ")), no active reaction or collision of this run " *
                "(active: $(join(sort!(collect(active_labels)), ", ")))"
        )
    )

    inactive = [process_label(p) => missing_species(registry, p) for p in catalog if !present(p)]
    warnings = assembly_warnings(registry, list, active_reactions, catalog)
    return SpeciesSystem(registry, active_reactions, active_collisions, AssemblyReport(inactive, warnings))
end

function rate_overrides(rates::Tuple)
    overrides = Dict{Symbol, AbstractRateModel}()
    for entry in rates
        (entry isa Pair && first(entry) isa Symbol && last(entry) isa AbstractRateModel) ||
            throw(ArgumentError("`config.rates` takes `label => rate model` pairs, got $(repr(entry))"))
        haskey(overrides, first(entry)) && throw(ArgumentError("`config.rates` gives :$(first(entry)) twice"))
        overrides[first(entry)] = last(entry)
    end
    return overrides
end

# The rate an active entry runs with, and its source: the override, or the catalog's resolved
active_rate(entry, overrides, registry, constants) =
    haskey(overrides, entry.label) ? (overrides[entry.label], "config.rates") :
    (resolve_rate(entry.rate, entry, registry, constants), rate_source(entry.rate))

"The labels of the species `process` involves that the registry lacks."
missing_species(reg::SpeciesRegistry, process::Process) =
    unique(Symbol[species_label(sp, reg.isotope) for sp in process_species(process) if !has_species(reg, sp)])

"""
    resolve_rate(rate, entry, registry, constants) -> AbstractRateModel

The rate model of `entry` (a reaction or a collision) for the registry's isotope, from its
declaration `rate` ([`RateData`](@ref) and its [`IsotopeRule`](@ref), or
[`LangevinCapture`](@ref)).
"""
resolve_rate(d::RateData{<:AbstractRateModel, SameForIsotopes}, _, ::SpeciesRegistry, ::PlasmaConstants) = d.model

function resolve_rate(d::RateData{<:AbstractRateModel, <:EqualVelocity}, entry, reg::SpeciesRegistry, constants::PlasmaConstants)
    reg.isotope === d.rule.reference && return d.model
    a, b = heavy_pair(entry)
    return TemperatureScaled(d.model, reduced_mass(a, b, d.rule.reference, constants) / reduced_mass(a, b, reg.isotope, constants))
end

function resolve_rate(d::RateData{<:AbstractRateModel, <:ReferenceOnly}, entry, reg::SpeciesRegistry, ::PlasmaConstants)
    reg.isotope === d.rule.reference && return d.model
    throw(
        ArgumentError(
            "$(entry.label): its rate ($(d.source)) holds for $(nameof(typeof(d.rule.reference))) alone and " *
                "declares no rule to carry it to $(nameof(typeof(reg.isotope))); give it in `config.rates`"
        )
    )
end

function resolve_rate(d::LangevinCapture, entry, reg::SpeciesRegistry, constants::PlasmaConstants{FT}) where {FT}
    μ = reduced_mass(heavy_pair(entry)..., reg.isotope, constants)
    return ConstantRate(FT(2π * sqrt(d.polarizability * constants.ee^2 / (4π * constants.eps0 * μ))))
end

function heavy_pair(r::Reaction)
    electron_count(r.reactants) == 0 ||
        throw(ArgumentError("$(r.label): electron impact has no heavy pair to carry a rate across isotopes by"))
    heavy = heavy_participants(r.reactants)
    return length(heavy) == 2 ? heavy : (only(heavy), only(heavy))
end
heavy_pair(x::Collision) = (x.ion, x.neutral)

function reduced_mass(a::AbstractSpecies, b::AbstractSpecies, iso::AbstractHydrogenIsotope, constants::PlasmaConstants)
    ma, mb = species_mass(a, iso, constants), species_mass(b, iso, constants)
    return ma * mb / (ma + mb)
end

# A main path of the default hydrogen list is off (a process lacking default hydrogen species
# alone, in a run that carries hydrogen), or nothing makes electrons
function assembly_warnings(reg::SpeciesRegistry, list::Tuple, active_reactions, catalog)
    warnings = String[]
    defaults = Set(map(sp -> species_label(sp, reg.isotope), DEFAULT_HYDROGEN_SPECIES))
    if any(sp -> sp.parent isa HydrogenParent, list)
        for process in catalog
            lacking = missing_species(reg, process)
            (!isempty(lacking) && lacking ⊆ defaults) || continue
            push!(warnings, "$(process_label(process)) ($(entry_summary(process, reg.isotope))) is off: the list lacks $(join(lacking, ", "))")
        end
    end
    any(a -> electron_change(a.reaction) > 0, active_reactions) ||
        push!(warnings, "no active reaction makes electrons")
    return warnings
end

# ── the report ──────────────────────────────────────────────────────────────────────────

participant_label(::Electron, ::AbstractHydrogenIsotope) = "e"
participant_label(sp::AbstractSpecies, iso::AbstractHydrogenIsotope) = string(species_label(sp, iso))

side_summary(side, iso) = join((n == 1 ? participant_label(x, iso) : "$n $(participant_label(x, iso))" for (x, n) in side), " + ")

entry_summary(r::Reaction, iso) = "$(side_summary(r.reactants, iso)) → $(side_summary(r.products, iso))"
entry_summary(x::Collision, iso) = "$(species_label(x.ion, iso)) on $(species_label(x.neutral, iso))"
entry_summary(p::ReactionProcess, iso) = let (reactants, products) = stoichiometry(p)
    "$(side_summary(reactants, iso)) → $(side_summary(products, iso))"
end
entry_summary(p::CollisionProcess, iso) = let (ion, neutral) = collision_pair(p)
    "$(species_label(ion, iso)) on $(species_label(neutral, iso))"
end

producer_summary(::ElectronSolve, _) = "electron solve"
producer_summary(p::LocalNetwork, iso) =
    "network, implicit $(species_label(p.implicit, iso))" * (isnothing(p.frozen) ? "" : ", frozen $(species_label(p.frozen, iso))")

closure_summary(::KineticClosure) = "kinetic"
closure_summary(::MaxwellianClosure) = "Maxwellian"
closure_summary(closure::RateClosure) = string(nameof(typeof(closure)))

"""
    species_report(system) -> String

The species system as text: the species with their columns and masses, the active reactions
and collisions with their producers, closures and rate sources, what is off and why, and the
warnings.
"""
function species_report(sys::SpeciesSystem)
    reg = sys.registry
    iso = reg.isotope
    io = IOBuffer()
    println(io, "Species (hydrogen: $(nameof(typeof(iso))))")
    for (k, r) in enumerate(reg.ions)
        println(io, "  ion $k      $(rpad(r.label, 8)) charge $(r.charge), mass $(r.mass) kg")
    end
    for (k, r) in enumerate(reg.neutrals)
        println(io, "  neutral $k  $(rpad(r.label, 8)) mass $(r.mass) kg")
    end
    println(io, "Reactions")
    for a in sys.reactions
        r = a.reaction
        println(
            io, "  $(rpad(r.label, 7)) $(rpad(entry_summary(r, iso), 34)) $(producer_summary(r.producer, iso)); ",
            "$(closure_summary(a.closure)), $(a.source)"
        )
    end
    println(io, "Collisions")
    for a in sys.collisions
        x = a.collision
        println(
            io, "  $(rpad(x.label, 7)) $(rpad(entry_summary(x, iso), 34)) $(nameof(typeof(x.moment))); ",
            "$(closure_summary(a.closure)), $(a.source)"
        )
    end
    isempty(sys.report.inactive) || println(io, "Off")
    for (label, lacking) in sys.report.inactive
        println(io, "  $(rpad(label, 7)) lacks $(join(lacking, ", "))")
    end
    isempty(sys.report.warnings) || println(io, "Warnings")
    for w in sys.report.warnings
        println(io, "  $w")
    end
    return String(take!(io))
end
