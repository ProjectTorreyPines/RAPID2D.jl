export Reaction, Collision, ElectronSolve, LocalNetwork, RateData, LangevinCapture
export SameForIsotopes, EqualVelocity, ReferenceOnly, ResonantChargeExchange, ElasticScattering
export LadderRates, ladder_rates

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
    ladder_rates(element, FT) -> LadderRates

The rates of `element`'s charge-state ladder in floating-point type `FT`. Every element defines
it.
"""
function ladder_rates end

"""
    ladder_reactions(element, rates, constants) -> Tuple

The reactions of `element`'s ladder, for every step z → z+1: the ionization `iz<X><z>` (counted
by the electron solve from the neutral, by the network above it) and the recombination
`rec<X><z+1>`, labelled by the parent's name (`izH0`, `recH1`).
"""
function ladder_reactions(p::Element, rates::LadderRates, constants::PlasmaConstants)
    Z = last(allowed_charges(p))
    length(rates.ionization) == length(rates.recombination) == length(rates.potentials_eV) == Z ||
        throw(ArgumentError("$(parent_name(p)): a ladder of $Z steps takes $Z rates of each kind and $Z potentials"))
    electron = Particles.electron
    name = parent_name(p)
    steps = ntuple(Z) do k
        lower, upper = p[k - 1], p[k]
        ionization = Reaction(
            Symbol(:iz, name, k - 1), (electron => 1, lower => 1) => (electron => 2, upper => 1);
            rate = RateData(rates.ionization[k], SameForIsotopes(), rates.sources[1]),
            producer = k == 1 ? ElectronSolve() : LocalNetwork(implicit = lower),
            energy = EventEnergy(rates.potentials_eV[k] * constants.ee),
        )
        recombination = Reaction(
            Symbol(:rec, name, k), (electron => 1, upper => 1) => (lower => 1,);
            rate = RateData(rates.recombination[k], SameForIsotopes(), rates.sources[2]),
            producer = LocalNetwork(implicit = upper),
            energy = CapturedElectron(),
        )
        (ionization, recombination)
    end
    return Tuple(Iterators.flatten(steps))
end

# The ladders of the elements a list carries
function element_ladders(list::Tuple, constants::PlasmaConstants{FT}) where {FT <: AbstractFloat}
    elements = unique(sp.parent for sp in list if sp.parent isa Element)
    return foldl(elements; init = ()) do ladders, p
        hasmethod(ladder_rates, Tuple{typeof(p), Type{FT}}) ||
            throw(ArgumentError("$(nameof(typeof(p))) is an element and defines no `ladder_rates`"))
        (ladders..., ladder_reactions(p, ladder_rates(p, FT), constants)...)
    end
end

# ── assembly ────────────────────────────────────────────────────────────────────────────

"""
    ActiveReaction(reaction, rate, source)

A reaction whose species the run all carries, with its rate model for the run's isotope and
where that rate comes from.
"""
struct ActiveReaction{R <: Reaction, M <: AbstractRateModel}
    reaction::R
    rate::M
    source::String
end

"A collision whose two species the run carries, with its rate for the run's isotope."
struct ActiveCollision{C <: Collision, M <: AbstractRateModel}
    collision::C
    rate::M
    source::String
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
    assemble_species_system(list, isotope, constants, eRRCs, iRRCs; rates = ()) -> SpeciesSystem

The species system of `list`: its registry, and the reactions and collisions of the catalog
(the hydrogen network and the ladder of every element listed) whose species it all carries,
their rates resolved for `isotope` or replaced by `rates` (`label => model` pairs). Throws for
an override that names no active reaction or collision, or data that cannot be carried to the
isotope.
"""
function assemble_species_system(
        list::Tuple, iso::AbstractHydrogenIsotope, constants::PlasmaConstants{FT}, eRRCs, iRRCs; rates::Tuple = ()
    ) where {FT <: AbstractFloat}
    registry = build_registry(list, iso, constants)
    reactions = (hydrogen_reactions(eRRCs, constants)..., element_ladders(list, constants)...)
    collisions = hydrogen_collisions(iRRCs, constants)
    labels = (map(r -> r.label, reactions)..., map(x -> x.label, collisions)...)
    allunique(labels) || throw(ArgumentError("two reactions or collisions share a label in ($(join(labels, ", ")))"))
    overrides = rate_overrides(rates)

    present(entry) = isempty(missing_species(registry, entry))
    activate(r::Reaction) = ActiveReaction(r, active_rate(r, overrides, registry, constants)...)
    activate(x::Collision) = ActiveCollision(x, active_rate(x, overrides, registry, constants)...)
    active_reactions = map(activate, filter(present, reactions))
    active_collisions = map(activate, filter(present, collisions))

    active_labels = Set{Symbol}(entry.label for entry in (reactions..., collisions...) if present(entry))
    unknown = setdiff(keys(overrides), active_labels)
    isempty(unknown) || throw(
        ArgumentError(
            "`config.rates` gives $(join(unknown, ", ")), no active reaction or collision of this run " *
                "(active: $(join(sort!(collect(active_labels)), ", ")))"
        )
    )

    inactive = [entry.label => missing_species(registry, entry) for entry in (reactions..., collisions...) if !present(entry)]
    warnings = assembly_warnings(registry, list, active_reactions, (reactions..., collisions...))
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

"The labels of the species `entry` involves that the registry lacks."
missing_species(reg::SpeciesRegistry, r::Reaction) = unique(
    Symbol[species_label(sp, reg.isotope) for sp in (heavy_participants(r.reactants)..., heavy_participants(r.products)...) if !has_species(reg, sp)]
)
missing_species(reg::SpeciesRegistry, x::Collision) =
    Symbol[species_label(sp, reg.isotope) for sp in (x.ion, x.neutral) if !has_species(reg, sp)]

"""
    resolve_rate(rate, entry, registry, constants) -> AbstractRateModel

The rate model of catalog entry `entry` for the registry's isotope, from its declaration `rate`
([`RateData`](@ref) and its [`IsotopeRule`](@ref), or [`LangevinCapture`](@ref)).
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

# A main path of the default hydrogen list is off (an entry lacking default hydrogen species
# alone, in a run that carries hydrogen), or nothing makes electrons
function assembly_warnings(reg::SpeciesRegistry, list::Tuple, active_reactions, entries)
    warnings = String[]
    defaults = Set(map(sp -> species_label(sp, reg.isotope), DEFAULT_HYDROGEN_SPECIES))
    if any(sp -> sp.parent isa HydrogenParent, list)
        for entry in entries
            lacking = missing_species(reg, entry)
            (!isempty(lacking) && lacking ⊆ defaults) || continue
            push!(warnings, "$(entry.label) ($(entry_summary(entry, reg.isotope))) is off: the list lacks $(join(lacking, ", "))")
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

producer_summary(::ElectronSolve, _) = "electron solve"
producer_summary(p::LocalNetwork, iso) =
    "network, implicit $(species_label(p.implicit, iso))" * (isnothing(p.frozen) ? "" : ", frozen $(species_label(p.frozen, iso))")

"""
    species_report(system) -> String

The species system as text: the species with their columns and masses, the active reactions
and collisions with their producers and rate sources, what is off and why, and the warnings.
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
        println(io, "  $(rpad(r.label, 7)) $(rpad(entry_summary(r, iso), 34)) $(producer_summary(r.producer, iso)); $(a.source)")
    end
    println(io, "Collisions")
    for a in sys.collisions
        x = a.collision
        println(io, "  $(rpad(x.label, 7)) $(rpad(entry_summary(x, iso), 34)) $(nameof(typeof(x.moment))); $(a.source)")
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
