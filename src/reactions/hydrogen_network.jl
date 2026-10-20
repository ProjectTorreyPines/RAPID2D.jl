# The hydrogen reactions and collisions as processes, with their data by closure and the sources
# of their rates (RRC_data/README.md tabulates them). Electron impact on H₂ has BREAK's kinetic,
# non-Maxwellian closure, the (E/p, Ē) surfaces. The other processes have the traditional
# Maxwellian closure only: the Janev–Reiter rate coefficients of HYDHEL section H.2 (EIRENE
# database, hydhel.tex), copied as printed, and the H₂⁺ (T, u_d) surfaces. HYDHEL gives each fit's
# lower bound; the upper bound 10⁴ eV is 2.1.8's, far above a start-up's temperatures.

"The default species list of a hydrogen run."
const DEFAULT_HYDROGEN_SPECIES = let H = Particles.H, H2 = Particles.H2
    (H2[0], H[0], H2[1], H[1])
end

const HYDHEL_2_1_5 = JanevFit(
    (
        -3.271396786375e+1, 1.353655609057e+1, -5.739328757388e+0, 1.563154982022e+0, -2.877056004391e-1,
        3.482559773737e-2, -2.63197617559e-3, 1.119543953861e-4, -2.039149852002e-6,
    ),
    (2.0, 1.0e4), "HYDHEL H.2 2.1.5, e + H(1s) → 2e + H⁺",
)

const HYDHEL_2_1_8 = JanevFit(
    (
        -2.95894689e+1, -0.52977995e+0, -0.00596813e+0, -0.0003354309e+0, -0.0030044468e+0,
        -0.0004702311e+0, 0.000251396e+0, -2.77628121e-5, 9.83367034e-7,
    ),
    (0.1, 1.0e4), "HYDHEL H.2 2.1.8, e + H⁺ → H(1s) + hν (into 1s only)",
)

const HYDHEL_2_2_11 = JanevFit(
    (
        -3.746192301092e+1, 1.559355031108e+1, -6.693238367093e+0, 1.981700292134e+0, -4.044820889297e-1,
        5.352391623039e-2, -4.317451841436e-3, 1.918499873454e-4, -3.591779705419e-6,
    ),
    (3.16, 1.0e4), "HYDHEL H.2 2.2.11, e + H₂⁺(v) → 2e + H⁺ + H⁺",
)

const HYDHEL_2_2_12 = JanevFit(
    (
        -1.781416067709e+1, 2.277799785711e+0, -1.266868411626e+0, 4.296170447419e-1, -9.609908013189e-2,
        1.387958040699e-2, -1.23134903947e-3, 6.042383126281e-5, -1.2475210409e-6,
    ),
    (0.2, 1.0e4), "HYDHEL H.2 2.2.12, e + H₂⁺(v) → e + H⁺ + H(1s)",
)

const HYDHEL_2_2_13 = JanevFit(
    (
        -3.408905929046e+1, 1.573560727511e+1, -6.992177456733e+0, 1.852216261706e+0, -3.130312806531e-1,
        3.383704123189e-2, -2.265770525273e-3, 8.565603779673e-5, -1.398131377085e-6,
    ),
    (2.0, 1.0e4), "HYDHEL H.2 2.2.13, e + H₂⁺(v) → e + H⁺ + H(n = 2)",
)

const HYDHEL_2_2_14 = JanevFit(
    (
        -1.670435653561e+1, -6.035644995682e-1, -1.942745783445e-8, -2.005952284492e-7, 2.962996104431e-8,
        2.134293274971e-8, -6.353973401838e-9, 6.152557460831e-10, -2.025361858319e-11,
    ),
    (0.1, 1.0e4), "HYDHEL H.2 2.2.14, e + H₂⁺(v) → H(1s) + H(n ≥ 2)",
)

const HYDHEL_2_2_15 = JanevFit(
    (
        -1.700270758355e+1, -4.050073042947e-1, 1.018733477232e-8, -1.695586285687e-8, 1.564311217508e-10,
        1.979725412288e-9, -4.395545994733e-10, 3.584926377078e-11, -1.024189019465e-12,
    ),
    (0.1, 1.0e4), "HYDHEL H.2 2.2.15, e + H₃⁺ → 3H or H₂ + H",
)

const HYDHEL_2_2_16 = JanevFit(
    (
        -3.078408636631e+1, 1.509421488513e+1, -7.349167207324e+0, 2.320966107642e+0, -4.818077551719e-1,
        6.389229162737e-2, -5.161880953089e-3, 2.303985092606e-4, -4.344846146197e-6,
    ),
    (1.26, 1.0e4), "HYDHEL H.2 2.2.16, e + H₃⁺ → e + H⁺ + 2H",
)

# Printed without a range; taken on the others' 0.1–10⁴ eV
const HYDHEL_3_1_8 = JanevFit(
    (
        -1.85028e+1, 3.708409e-1, 7.949876e-3, -6.143769e-4, -4.698969e-4,
        -4.096807e-4, 1.440382e-4, -1.514243e-5, 5.122435e-7,
    ),
    (0.1, 1.0e4), "HYDHEL H.2 3.1.8, p + H(1s) → H(1s) + p (vs. T_p, H at rest)",
)

# The electrons' loss in the H₂⁺ and H₃⁺ excitations and ionization: the HYDHEL H.1 thresholds
# (lowest over v = 0…9). The mean loss is larger (11.8 eV vertical 1sσg → 2pσu at v = 0).
const E_TH_2_2_11_EV = 14.7
const E_TH_2_2_12_EV = 2.4
const E_TH_2_2_13_EV = 14.0
const E_TH_2_2_16_EV = 14.0

"Static polarizability volume of H₂ [m³]: 0.8045 Å³ (CRC Handbook, atomic and molecular polarizabilities)."
const H2_POLARIZABILITY = 0.8045e-30

"Branches of H₃⁺ dissociative recombination: 3H 0.75, H₂ + H 0.25 (Datz et al., PRL 74, 896 (1995))."
const H3_DR_THREE_BODY = 0.75

"""
    ladder_rates(::Hydrogen, ::MaxwellianClosure, FT)

H's single step: ionization from the ground state (HYDHEL 2.1.5) and radiative recombination
into 1s (HYDHEL 2.1.8, a lower bound of the total), ionization potential 13.598 eV.
"""
ladder_rates(::Hydrogen, ::MaxwellianClosure, ::Type{FT}) where {FT <: AbstractFloat} = LadderRates(
    (LogPolynomialRate(FT, HYDHEL_2_1_5),), (LogPolynomialRate(FT, HYDHEL_2_1_8),), (FT(13.598),),
    (HYDHEL_2_1_5.source, HYDHEL_2_1_8.source),
)

# The data each closure has for a process: electron impact on H₂ from the kinetic (E/p, Ē) tables,
# the rest from Maxwellian fits and tables; a rate or energy for another closure is one more method
electron_table(sources::RateSources, name::Symbol) =
    RateData(TabulatedRate(getfield(sources.eRRCs, name)), SameForIsotopes(), "eRRCs_EoverP_Erg.h5 $name")
ion_table(sources::RateSources, name::Symbol) =
    RateData(TabulatedRate(getfield(sources.iRRCs, name)), EqualVelocity(Protium()), "iRRCs_T_ud.h5 $name")
electron_fit(sources::RateSources, fit::JanevFit; fraction = 1) =
    RateData(LogPolynomialRate(float_type(sources), fit; fraction), SameForIsotopes(), fit.source)
threshold_energy(sources::RateSources, E_th_eV) = EventEnergy(float_type(sources)(E_th_eV) * sources.constants.ee)

# ── electron impact on H₂ ───────────────────────────────────────────────────────────────

"e + H2[0] → 2e + H2[1] (`iz`)."
struct H2Ionization <: ReactionProcess end
process_label(::H2Ionization) = :iz
stoichiometry(::H2Ionization) =
    (Particles.electron => 1, Particles.H2[0] => 1) => (Particles.electron => 2, Particles.H2[1] => 1)
producer(::H2Ionization) = ElectronSolve()
rate_data(::H2Ionization, ::KineticClosure, sources::RateSources) = electron_table(sources, :K_iz)
electron_energy(::H2Ionization, ::RateClosure, sources::RateSources) =
    EventEnergy(sources.constants.iz_erg_eV * sources.constants.ee)

"e + H2[0] → 2e + H[1] + H[0] (`diz`)."
struct H2DissociativeIonization <: ReactionProcess end
process_label(::H2DissociativeIonization) = :diz
stoichiometry(::H2DissociativeIonization) =
    (Particles.electron => 1, Particles.H2[0] => 1) => (Particles.electron => 2, Particles.H[1] => 1, Particles.H[0] => 1)
producer(::H2DissociativeIonization) = ElectronSolve()
rate_data(::H2DissociativeIonization, ::KineticClosure, sources::RateSources) = electron_table(sources, :K_diss_iz)
electron_energy(::H2DissociativeIonization, ::RateClosure, sources::RateSources) =
    EventEnergy(sources.constants.diss_iz_erg_eV * sources.constants.ee)

"e + H2[0] → e + 2 H[0] (`dex`): the triplets and the dissociating fraction of the B, C singlets."
struct H2DissociativeExcitation <: ReactionProcess end
process_label(::H2DissociativeExcitation) = :dex
stoichiometry(::H2DissociativeExcitation) =
    (Particles.electron => 1, Particles.H2[0] => 1) => (Particles.electron => 1, Particles.H[0] => 2)
producer(::H2DissociativeExcitation) = ElectronSolve()
rate_data(::H2DissociativeExcitation, ::KineticClosure, sources::RateSources) = electron_table(sources, :K_diss_exc)
electron_energy(::H2DissociativeExcitation, ::KineticClosure, sources::RateSources) =
    ContinuousPower(TabulatedRate(sources.eRRCs.Kerg_diss_exc))

# ── electron impact on H₂⁺ ──────────────────────────────────────────────────────────────

"e + H2[1] → e + H[1] + H[0] through 2pσᵤ, H in 1s (`de2`)."
struct H2IonDissociativeExcitation <: ReactionProcess end
process_label(::H2IonDissociativeExcitation) = :de2
stoichiometry(::H2IonDissociativeExcitation) =
    (Particles.electron => 1, Particles.H2[1] => 1) => (Particles.electron => 1, Particles.H[1] => 1, Particles.H[0] => 1)
producer(::H2IonDissociativeExcitation) = LocalNetwork(implicit = Particles.H2[1])
rate_data(::H2IonDissociativeExcitation, ::MaxwellianClosure, sources::RateSources) = electron_fit(sources, HYDHEL_2_2_12)
electron_energy(::H2IonDissociativeExcitation, ::MaxwellianClosure, sources::RateSources) =
    threshold_energy(sources, E_TH_2_2_12_EV)

"e + H2[1] → e + H[1] + H[0] through 2pπᵤ, H in n = 2 (`de2b`)."
struct H2IonDissociativeExcitationExcitedH <: ReactionProcess end
process_label(::H2IonDissociativeExcitationExcitedH) = :de2b
stoichiometry(::H2IonDissociativeExcitationExcitedH) =
    (Particles.electron => 1, Particles.H2[1] => 1) => (Particles.electron => 1, Particles.H[1] => 1, Particles.H[0] => 1)
producer(::H2IonDissociativeExcitationExcitedH) = LocalNetwork(implicit = Particles.H2[1])
rate_data(::H2IonDissociativeExcitationExcitedH, ::MaxwellianClosure, sources::RateSources) = electron_fit(sources, HYDHEL_2_2_13)
electron_energy(::H2IonDissociativeExcitationExcitedH, ::MaxwellianClosure, sources::RateSources) =
    threshold_energy(sources, E_TH_2_2_13_EV)

"e + H2[1] → 2e + 2 H[1] (`di2`)."
struct H2IonDissociativeIonization <: ReactionProcess end
process_label(::H2IonDissociativeIonization) = :di2
stoichiometry(::H2IonDissociativeIonization) =
    (Particles.electron => 1, Particles.H2[1] => 1) => (Particles.electron => 2, Particles.H[1] => 2)
producer(::H2IonDissociativeIonization) = LocalNetwork(implicit = Particles.H2[1])
rate_data(::H2IonDissociativeIonization, ::MaxwellianClosure, sources::RateSources) = electron_fit(sources, HYDHEL_2_2_11)
electron_energy(::H2IonDissociativeIonization, ::MaxwellianClosure, sources::RateSources) =
    threshold_energy(sources, E_TH_2_2_11_EV)

"e + H2[1] → 2 H[0] (`dr2`)."
struct H2IonDissociativeRecombination <: ReactionProcess end
process_label(::H2IonDissociativeRecombination) = :dr2
stoichiometry(::H2IonDissociativeRecombination) = (Particles.electron => 1, Particles.H2[1] => 1) => (Particles.H[0] => 2,)
producer(::H2IonDissociativeRecombination) = LocalNetwork(implicit = Particles.H2[1])
rate_data(::H2IonDissociativeRecombination, ::MaxwellianClosure, sources::RateSources) = electron_fit(sources, HYDHEL_2_2_14)
electron_energy(::H2IonDissociativeRecombination, ::MaxwellianClosure, ::RateSources) = CapturedElectron()

# ── H₃⁺ ─────────────────────────────────────────────────────────────────────────────────

"H2[1] + H2[0] → H3[1] + H[0] (`h3`), at the Langevin rate: independent of the distribution."
struct H3IonFormation <: ReactionProcess end
process_label(::H3IonFormation) = :h3
stoichiometry(::H3IonFormation) =
    (Particles.H2[1] => 1, Particles.H2[0] => 1) => (Particles.H3[1] => 1, Particles.H[0] => 1)
producer(::H3IonFormation) = LocalNetwork(implicit = Particles.H2[1], frozen = Particles.H2[0])
rate_data(::H3IonFormation, ::RateClosure, ::RateSources) =
    LangevinCapture(H2_POLARIZABILITY, "Langevin capture on H₂ (polarizability 0.8045 Å³)")
electron_energy(::H3IonFormation, ::RateClosure, ::RateSources) = nothing

"e + H3[1] → H2[0] + H[0] (`dr3a`), a quarter of H₃⁺ dissociative recombination."
struct H3IonRecombinationToH2 <: ReactionProcess end
process_label(::H3IonRecombinationToH2) = :dr3a
stoichiometry(::H3IonRecombinationToH2) =
    (Particles.electron => 1, Particles.H3[1] => 1) => (Particles.H2[0] => 1, Particles.H[0] => 1)
producer(::H3IonRecombinationToH2) = LocalNetwork(implicit = Particles.H3[1])
rate_data(::H3IonRecombinationToH2, ::MaxwellianClosure, sources::RateSources) =
    electron_fit(sources, HYDHEL_2_2_15; fraction = 1 - H3_DR_THREE_BODY)
electron_energy(::H3IonRecombinationToH2, ::MaxwellianClosure, ::RateSources) = CapturedElectron()

"e + H3[1] → 3 H[0] (`dr3b`), three quarters of H₃⁺ dissociative recombination."
struct H3IonRecombinationToH <: ReactionProcess end
process_label(::H3IonRecombinationToH) = :dr3b
stoichiometry(::H3IonRecombinationToH) = (Particles.electron => 1, Particles.H3[1] => 1) => (Particles.H[0] => 3,)
producer(::H3IonRecombinationToH) = LocalNetwork(implicit = Particles.H3[1])
rate_data(::H3IonRecombinationToH, ::MaxwellianClosure, sources::RateSources) =
    electron_fit(sources, HYDHEL_2_2_15; fraction = H3_DR_THREE_BODY)
electron_energy(::H3IonRecombinationToH, ::MaxwellianClosure, ::RateSources) = CapturedElectron()

"e + H3[1] → e + H[1] + 2 H[0] (`de3`)."
struct H3IonDissociativeExcitation <: ReactionProcess end
process_label(::H3IonDissociativeExcitation) = :de3
stoichiometry(::H3IonDissociativeExcitation) =
    (Particles.electron => 1, Particles.H3[1] => 1) => (Particles.electron => 1, Particles.H[1] => 1, Particles.H[0] => 2)
producer(::H3IonDissociativeExcitation) = LocalNetwork(implicit = Particles.H3[1])
rate_data(::H3IonDissociativeExcitation, ::MaxwellianClosure, sources::RateSources) = electron_fit(sources, HYDHEL_2_2_16)
electron_energy(::H3IonDissociativeExcitation, ::MaxwellianClosure, sources::RateSources) =
    threshold_energy(sources, E_TH_2_2_16_EV)

# ── ion–neutral collisions ──────────────────────────────────────────────────────────────

"H[1] on H[0], resonant charge exchange (`cxH`)."
struct ProtonChargeExchange <: CollisionProcess end
process_label(::ProtonChargeExchange) = :cxH
collision_pair(::ProtonChargeExchange) = (Particles.H[1], Particles.H[0])
collision_moment(::ProtonChargeExchange) = ResonantChargeExchange()
rate_data(::ProtonChargeExchange, ::MaxwellianClosure, sources::RateSources) =
    RateData(LogPolynomialRate(float_type(sources), HYDHEL_3_1_8), EqualVelocity(Protium()), HYDHEL_3_1_8.source)

"H2[1] on H2[0], elastic scattering (`elH2`)."
struct H2IonElasticScattering <: CollisionProcess end
process_label(::H2IonElasticScattering) = :elH2
collision_pair(::H2IonElasticScattering) = (Particles.H2[1], Particles.H2[0])
collision_moment(::H2IonElasticScattering) = ElasticScattering()
rate_data(::H2IonElasticScattering, ::MaxwellianClosure, sources::RateSources) = ion_table(sources, :Elastic)

"H2[1] on H2[0], resonant charge exchange (`cxH2`)."
struct H2IonChargeExchange <: CollisionProcess end
process_label(::H2IonChargeExchange) = :cxH2
collision_pair(::H2IonChargeExchange) = (Particles.H2[1], Particles.H2[0])
collision_moment(::H2IonChargeExchange) = ResonantChargeExchange()
rate_data(::H2IonChargeExchange, ::MaxwellianClosure, sources::RateSources) = ion_table(sources, :Charge_Exchange)

"""
    HYDROGEN_PROCESSES

The hydrogen reactions and collisions other than H's ladder (which comes with `H` in the list),
active in a run that carries all their species.
"""
const HYDROGEN_PROCESSES = (
    H2Ionization(), H2DissociativeIonization(), H2DissociativeExcitation(),
    H2IonDissociativeExcitation(), H2IonDissociativeExcitationExcitedH(), H2IonDissociativeIonization(),
    H2IonDissociativeRecombination(),
    H3IonFormation(), H3IonRecombinationToH2(), H3IonRecombinationToH(), H3IonDissociativeExcitation(),
    ProtonChargeExchange(), H2IonElasticScattering(), H2IonChargeExchange(),
)
