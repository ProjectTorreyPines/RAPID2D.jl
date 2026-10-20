# The hydrogen reactions and collisions, with the sources of their rates (RRC_data/README.md
# tabulates them). The tables are the electron (E/p, Ē) surfaces and the H₂⁺ (T, u_d) surfaces of
# RRC_data; the fits are the Janev–Reiter Maxwellian rate coefficients of HYDHEL, section H.2
# (EIRENE database, hydhel.tex), copied as printed. HYDHEL gives each fit's lower bound; the upper
# bound 10⁴ eV is 2.1.8's, far above a start-up's temperatures.

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
const E_TH_2_2_16_EV = 14.0

"Static polarizability volume of H₂ [m³]: 0.8045 Å³ (CRC Handbook, atomic and molecular polarizabilities)."
const H2_POLARIZABILITY = 0.8045e-30

"Branches of H₃⁺ dissociative recombination: 3H 0.75, H₂ + H 0.25 (Datz et al., PRL 74, 896 (1995))."
const H3_DR_THREE_BODY = 0.75

"""
    ladder_rates(::Hydrogen, FT)

H's single step: ionization from the ground state (HYDHEL 2.1.5) and radiative recombination
into 1s (HYDHEL 2.1.8, a lower bound of the total), ionization potential 13.598 eV.
"""
ladder_rates(::Hydrogen, ::Type{FT}) where {FT <: AbstractFloat} = LadderRates(
    (LogPolynomialRate(FT, HYDHEL_2_1_5),), (LogPolynomialRate(FT, HYDHEL_2_1_8),), (FT(13.598),),
    (HYDHEL_2_1_5.source, HYDHEL_2_1_8.source),
)

"""
    hydrogen_reactions(eRRCs, constants) -> Tuple

The reactions among the hydrogen species other than H's ladder ([`ladder_rates`](@ref)):
electron impact on H₂ from the electron tables, on H₂⁺ and H₃⁺ from HYDHEL, and H₃⁺ formation
at the Langevin rate.
"""
function hydrogen_reactions(eRRCs::Electron_RRCs, c::PlasmaConstants{FT}) where {FT <: AbstractFloat}
    e, H, H2, H3 = Particles.electron, Particles.H, Particles.H2, Particles.H3
    table(name) = RateData(TabulatedRate(getfield(eRRCs, name)), SameForIsotopes(), "eRRCs_EoverP_Erg.h5 $name")
    fit(data; fraction = 1) = RateData(LogPolynomialRate(FT, data; fraction), SameForIsotopes(), data.source)
    eV = c.ee
    return (
        Reaction(
            :iz, (e => 1, H2[0] => 1) => (e => 2, H2[1] => 1);
            rate = table(:K_iz), producer = ElectronSolve(), energy = EventEnergy(c.iz_erg_eV * eV),
        ),
        Reaction(
            :diz, (e => 1, H2[0] => 1) => (e => 2, H[1] => 1, H[0] => 1);
            rate = table(:K_diss_iz), producer = ElectronSolve(), energy = EventEnergy(c.diss_iz_erg_eV * eV),
        ),
        Reaction(
            :dex, (e => 1, H2[0] => 1) => (e => 1, H[0] => 2);
            rate = table(:K_diss_exc), producer = ElectronSolve(), energy = ContinuousPower(TabulatedRate(eRRCs.Kerg_diss_exc)),
        ),
        Reaction(
            :de2, (e => 1, H2[1] => 1) => (e => 1, H[1] => 1, H[0] => 1);
            rate = fit(HYDHEL_2_2_12), producer = LocalNetwork(implicit = H2[1]), energy = EventEnergy(FT(E_TH_2_2_12_EV) * eV),
        ),
        Reaction(
            :di2, (e => 1, H2[1] => 1) => (e => 2, H[1] => 2);
            rate = fit(HYDHEL_2_2_11), producer = LocalNetwork(implicit = H2[1]), energy = EventEnergy(FT(E_TH_2_2_11_EV) * eV),
        ),
        Reaction(
            :dr2, (e => 1, H2[1] => 1) => (H[0] => 2,);
            rate = fit(HYDHEL_2_2_14), producer = LocalNetwork(implicit = H2[1]), energy = CapturedElectron(),
        ),
        Reaction(
            :h3, (H2[1] => 1, H2[0] => 1) => (H3[1] => 1, H[0] => 1);
            rate = LangevinCapture(H2_POLARIZABILITY, "Langevin capture on H₂ (polarizability 0.8045 Å³)"),
            producer = LocalNetwork(implicit = H2[1], frozen = H2[0]),
        ),
        Reaction(
            :dr3a, (e => 1, H3[1] => 1) => (H2[0] => 1, H[0] => 1);
            rate = fit(HYDHEL_2_2_15; fraction = 1 - H3_DR_THREE_BODY), producer = LocalNetwork(implicit = H3[1]),
            energy = CapturedElectron(),
        ),
        Reaction(
            :dr3b, (e => 1, H3[1] => 1) => (H[0] => 3,);
            rate = fit(HYDHEL_2_2_15; fraction = H3_DR_THREE_BODY), producer = LocalNetwork(implicit = H3[1]),
            energy = CapturedElectron(),
        ),
        Reaction(
            :de3, (e => 1, H3[1] => 1) => (e => 1, H[1] => 1, H[0] => 2);
            rate = fit(HYDHEL_2_2_16), producer = LocalNetwork(implicit = H3[1]), energy = EventEnergy(FT(E_TH_2_2_16_EV) * eV),
        ),
    )
end

"""
    hydrogen_collisions(iRRCs, constants) -> Tuple

The ion–neutral collisions among the hydrogen species: H⁺ on H by resonant charge exchange
(HYDHEL 3.1.8), H₂⁺ on H₂ by elastic scattering and charge exchange (the H₂⁺ tables). The data are
protium's, carried to the other isotopes at equal velocity.
"""
function hydrogen_collisions(iRRCs::H2_Ion_RRCs, ::PlasmaConstants{FT}) where {FT <: AbstractFloat}
    H, H2 = Particles.H, Particles.H2
    table(name) = RateData(TabulatedRate(getfield(iRRCs, name)), EqualVelocity(Protium()), "iRRCs_T_ud.h5 $name")
    return (
        Collision(
            :cxH, H[1], H[0];
            rate = RateData(LogPolynomialRate(FT, HYDHEL_3_1_8), EqualVelocity(Protium()), HYDHEL_3_1_8.source),
            moment = ResonantChargeExchange(),
        ),
        Collision(:elH2, H2[1], H2[0]; rate = table(:Elastic), moment = ElasticScattering()),
        Collision(:cxH2, H2[1], H2[0]; rate = table(:Charge_Exchange), moment = ResonantChargeExchange()),
    )
end
