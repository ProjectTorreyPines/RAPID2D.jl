# Reaction-rate coefficient tables

Maxwellian-averaged rate coefficients `⟨σv⟩` [m³/s] for hydrogen breakdown, on regular
grids in HDF5.

| file | axes | contents |
|---|---|---|
| `eRRCs_EoverP_Erg.h5` | `EoverP` × `Erg_eV`, 201 × 300 | electron `Ionization`, `Total_Momentum`, `Momentum_by_ela`, `Total_Excitation` |
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

## Checking a table

`scripts/check_ion_rrc_tables.jl` recomputes `⟨σv⟩(T, u_d)` for the H₂⁺ channels by
integrating the source cross sections over a shifted Maxwellian, and compares point by
point. Run it against any regenerated file:

```
julia --project=. scripts/check_ion_rrc_tables.jl RRC_data/iRRCs_T_ud.h5
```
