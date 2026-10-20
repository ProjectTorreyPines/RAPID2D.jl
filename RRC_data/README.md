# Reaction-rate coefficient tables

Maxwellian-averaged rate coefficients `⟨σv⟩` [m³/s] for hydrogen breakdown, on regular
grids in HDF5.

| file | axes | contents |
|---|---|---|
| `eRRCs_EoverP_Erg.h5` | `EoverP` × `Erg_eV`, 201 × 300 | electron ledgers, 2026-08 group schema, 19 datasets and no legacy surface: axes `EoverP`, `Erg_eV`; particles `K_iz`, `K_diss_iz`, `K_exc`, `K_diss_exc`, `K_ela`; momentum `K_mom`, `K_mom_by_ela`, `K_mom_by_exc`, `K_mom_by_diss_exc`, `K_mom_by_iz`, `K_mom_by_diss_iz`; energy `L_ela`, `L_exc`, `L_diss_exc`, `L_tot` [W·m³]; plus `characteristic_exc_erg_eV`, `saturation`. The pre-2026-08 aliases (`Ionization`, `Total_Momentum`, `Elastic`, `Momentum_by_*`, `Total_Excitation`) no longer ship: `Total_Excitation` was corrupt in the cold band (see `internal/docs/src/notes/design/bd-reaction-schema-migration.md` §7.1b), and the legacy build is retired rather than kept around unread |
| `eRRCs_T_ud.h5` | `T_eV` × `ud_para`, 200 × 200 | electron `Halpha`, `Ionization`, `Momentum`, `Total_Excitation`, `Dissoc_Ionz`, `Recomb_H2Ion`, `Recomb_H3Ion` |
| `iRRCs_T_ud.h5` | `T_eV` × `ud_para`, 100 × 100 | H₂⁺ `Elastic`, `Charge_Exchange`, `Particle_Exchange`, `Target_Ionization`, `Projectile_Dissociation` |

The `(T, u_d)` families average a **drifting** Maxwellian at temperature `T` and parallel
drift `u_d` against a target at rest. `eRRCs_EoverP_Erg.h5` averages a Boltzmann-solver
EEDF parameterised by reduced field `E/p` and mean energy `Ē`.

## ⚠️ Axis order in the `(T, u_d)` files

**Both are stored transposed** — index 1 is drift, index 2 is temperature — while the code
that reads them indexes `[temperature, drift]`. Both are square, so nothing detects the
mismatch: it silently returns the right number at the wrong point.

A file therefore declares its own layout in a root attribute, and the reader follows it:

| `axis_order` | meaning |
|---|---|
| `"T_eV,ud_para"` | index 1 is temperature |
| `"ud_para,T_eV"` | index 1 is drift |
| absent | assumed `"ud_para,T_eV"` — the two files above carry no attribute |

**Stamp any new or regenerated file.** Without the attribute it is read as transposed.

## ⚠️ Suppressed low-energy physics

The source cross sections return 0 outside their fitted range, which for these channels
suppresses real reactions rather than absent ones:

| channel | guard | effect at `T_i` ≈ 0.2 eV |
|---|---|---|
| `Charge_Exchange` | `E < 4.9 eV → 0` | ~10⁷ × too small — symmetric resonant CX has no threshold and is largest at low velocity |
| `Particle_Exchange` | `E < 1.64 eV → 0` | ~400 × too small — measured to proceed at the Langevin rate far below this |

`Charge_Exchange` is the one that reaches transport. Treat both as unreliable below a few
eV pending a cross-section revision.

## Reading the 2026-08 electron table

Three ledgers, three weightings of the same `⟨σv⟩` average — see
`internal/docs/src/notes/design/bd-2026-08-governing-equations.md` §2. In short:

- `K_*` [m³/s] count events. Consume as `ν = n_gas·K`.
- `L_*` [W·m³] carry energy. Consume as `P = n_gas·L` [W per electron]. RAPID2D renames
  these to `Kerg_*` on read, because `L_` is already *length* in this codebase.
- `K_mom*` [m³/s] is a `v_z`-weighted drift-friction moment, NOT the density-weighted
  collision frequency. `K_ela` is neither — it is the total elastic collision frequency,
  for dt sizing, and RAPID2D does not read it.

`L_ela` already contains `2mₑ/M_H2` and the `e`. Do not apply them again.

Never form `L_x/K_x` at runtime. The pair is deliberately asymmetric for the DISS group and
is 0/0 on most of the grid; both ledgers must be read directly.

### ⚠️ RAPID2D deviates from `L_exc`'s `consume_as` contract

`L_exc`'s `consume_as` attribute says `P = n_e * n_gas * L_exc`, no correction factor —
and every other ledger column here is consumed exactly as its attribute says. `L_exc` is
the one exception: RAPID2D multiplies it by the same cold-target factor applied to
`L_ela`, `(1 - (3/2)·T_gas/Ē)` (see `update_electron_heating_powers!` in
`src/physics/physics.jl`).

Reason: `K_exc`/`L_exc`, like the elastic surfaces, are one-way coefficients — computed
against a stationary, ground-state H2 — with no superelastic return channel. That is
correct for vibration (0.516 eV) and the electronic channels (>= 11.2 eV), which have no
thermal population at 300 K, but not for rotation: `ΔE_rot = 0.0441 eV` is only 1.7x room
temperature, so a real fraction of molecules are already rotationally excited and handing
energy back. Left uncorrected, `L_exc` keeps draining electron energy even at `Tₑ =
T_gas`, which is a real, measured effect in this codebase (electrons settle at ~0.0111 eV
instead of relaxing to T_gas in `Atomic_Collision`-only relaxation).

The correction is phenomenological, not exact detailed balance, because rotation is not
exported as its own dataset — it is folded into the whole EXC group. The exact rotational
factor, `1 - exp(ΔE/Tₑ - ΔE/T_gas)`, does not return to 1 at high `Tₑ`; applied to the
whole group it would suppress electronic excitation by a permanent 18.4%. The linear
cold-target form shares the exact factor's two limits (0 at `Tₑ = T_gas`, → 1 at high
`Tₑ`) without that permanent cost.

Retire this override the day an `L_exc_rot` dataset lets the exact factor be applied to
the rotational part alone, leaving the rest of `EXC` on the literal `consume_as`
contract.

## Checking a table

`scripts/check_ion_rrc_tables.jl` recomputes `⟨σv⟩(T, u_d)` for the H₂⁺ channels by
integrating the source cross sections over a shifted Maxwellian, and compares point by
point. Run it against any regenerated file:

```
julia --project=. scripts/check_ion_rrc_tables.jl RRC_data/iRRCs_T_ud.h5
```

## Rates of the species catalog

`config.species` assembles a catalog of reactions and collisions (`src/reactions/hydrogen_network.jl`).
Their rates come from the tables above and from fits.

The fits are the Janev–Reiter Maxwellian rate coefficients of the EIRENE databases, section H.2,
copied as printed; the code adds $\ln 10^{-6}$ to the first coefficient for m³/s:

$$
\ln \langle \sigma v \rangle \left[ \mathrm{cm^3/s} \right] = \sum_{n=0}^{8} b_n \left( \ln T \left[ \mathrm{eV} \right] \right)^n
$$

The source is `hydhel.tex` of the EIRENE repository (`Database/AMdata`, git f8f63fa0, 2026-04-15).
HYDHEL prints each fit's lower bound; the upper bound, 10⁴ eV, is the one 2.1.8 prints. Outside
its range a fit is frozen and its derivative is zero. `species_catalog_test.jl` checks the copied
coefficients against the rate HYDHEL prints at each lower bound (6 %) and at each maximum (2 %).

| label | reaction | rate | range [eV] | electron energy per event |
|---|---|---|---|---|
| `iz` | e + H2[0] → 2e + H2[1] | `K_iz` (E/p, Ē) | table | 15.426 eV (`iz_erg_eV`) |
| `diz` | e + H2[0] → 2e + H[1] + H[0] | `K_diss_iz` (E/p, Ē) | table | 35 eV (`diss_iz_erg_eV`) |
| `dex` | e + H2[0] → e + 2 H[0] | `K_diss_exc` (E/p, Ē) | table | power `Kerg_diss_exc` |
| `izH0` | e + H[0] → 2e + H[1] | HYDHEL 2.1.5 (ground state) | 2 – 10⁴ | 13.598 eV |
| `recH1` | e + H[1] → H[0] | HYDHEL 2.1.8 (into 1s only) | 0.1 – 10⁴ | captured electron |
| `de2` | e + H2[1] → e + H[1] + H[0] | HYDHEL 2.2.12 | 0.2 – 10⁴ | 2.4 eV (H.1 threshold) |
| `di2` | e + H2[1] → 2e + 2 H[1] | HYDHEL 2.2.11 | 3.16 – 10⁴ | 14.7 eV (H.1 threshold) |
| `dr2` | e + H2[1] → 2 H[0] | HYDHEL 2.2.14 | 0.1 – 10⁴ | captured electron |
| `h3` | H2[1] + H2[0] → H3[1] + H[0] | Langevin, polarizability of H₂ 0.8045 Å³ | — | — |
| `dr3a` | e + H3[1] → H2[0] + H[0] | HYDHEL 2.2.15 × 0.25 | 0.1 – 10⁴ | captured electron |
| `dr3b` | e + H3[1] → 3 H[0] | HYDHEL 2.2.15 × 0.75 | 0.1 – 10⁴ | captured electron |
| `de3` | e + H3[1] → e + H[1] + 2 H[0] | HYDHEL 2.2.16 | 1.26 – 10⁴ | 14.0 eV (H.1 threshold) |

| label | collision | rate | moment |
|---|---|---|---|
| `cxH` | H[1] on H[0] | HYDHEL 3.1.8 (vs. the proton temperature, H at rest; no range printed, 0.1 – 10⁴ used) | resonant charge exchange |
| `elH2` | H2[1] on H2[0] | `iRRCs_T_ud.h5` `Elastic` | elastic, the fraction μ/mᵢ of the momentum |
| `cxH2` | H2[1] on H2[0] | `iRRCs_T_ud.h5` `Charge_Exchange` | resonant charge exchange |

A captured electron (a recombination) takes its kinetic energy out of the electron fluid. For
Maxwellian electrons on a target at rest this follows from the rate itself, the H.8–H.2 relation
of AMJUEL §1.3.8:

$$
\frac{\langle \sigma v E \rangle}{\langle \sigma v \rangle} = e T_e \left( \frac{3}{2} + \frac{d \ln K}{d \ln T_e} \right)
$$

For 2.2.14 it is 0.896 times the electron temperature, as AMJUEL's H.8 2.2.14 fit gives.

H₃⁺ forms at the Langevin capture rate, which carries isotopes through the reduced mass μ:

$$
K = 2 \pi \sqrt{\frac{\alpha e^2}{4 \pi \varepsilon_0 \mu}}
$$

With the CRC Handbook polarizability of H₂ it is 2.09×10⁻⁹ cm³/s for H₂⁺ + H₂, near the measured
2.1×10⁻⁹ cm³/s. H₃⁺ dissociative recombination branches 0.75 into 3H and 0.25 into H₂ + H (Datz
et al., PRL 74, 896 (1995)).

Electron-impact rates are shared by the isotopes (their vibrational dependence aside). The
heavy-pair data are protium's and carry to another isotope at equal velocity, exact for resonant
charge exchange and approximate for the elastic table:

$$
K(T) = K_\mathrm{H} \left( T \frac{\mu_\mathrm{H}}{\mu} \right)
$$

Data declared `ReferenceOnly` refuse another isotope until `config.rates` gives the rate.

### Known gaps

- `de2`, `di2`, `de3` charge the electrons the lowest threshold over v = 0…9. The mean loss is
  larger: the vertical 1sσg → 2pσu transition of H₂⁺ at v = 0 is 11.8 eV. The fragments'
  Franck–Condon energies are not in the catalog yet. Both belong with the energy ledger.
- `izH0` has no excitation radiation of H(n ≥ 2), and `recH1` recombines into 1s alone, a lower
  bound of the total. The collisional–radiative rates (AMJUEL H.4 and H.10, 2.1.5 and 2.1.8) are
  the next step.
- Collisions of H[1] on H2[0] and of H3[1] on H2[0] are not in the catalog yet.
