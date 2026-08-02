# Neutral H2 fill: kinetic diffusivity.
#
# A molecule's free path is terminated by whichever happens first — a collision
# with another molecule, destruction by electron impact, or arrival at the wall.
# The three are summed as rates (Matthiessen), so the shortest one dominates:
#
#     1/λ = √2·n_gas·σ  +  ν_iz/v_th  +  1/L
#            ‾‾‾‾‾‾‾‾‾     ‾‾‾‾‾‾‾‾‾     ‾‾‾
#            gas-gas       ionization    wall (Knudsen)
#
#     D = ½·v_th·λ
#
# **Why the wall term is not optional.** At tokamak breakdown pressures the
# gas-gas path is METRES: 4.2 m at 2e-5 Torr, 2.8 m at 4 mPa, against a vessel of
# order 1 m. The flow there is free-molecular, not diffusive, and an uncapped D
# makes the diffusive crossing time L²/D fall BELOW the ballistic floor L/v_th —
# transport faster than free streaming, which is impossible. Counting the wall as
# a collision channel recovers Knudsen diffusion, D → ½·v_th·L, i.e. crossing at
# roughly the thermal transit rate. It also removes the n_gas → 0 singularity on
# burnt-out cells without any NaN scrubbing.
#
# The MATLAB original (`c_RAPID.m`, `Update_n_H2_gas_density`) has a CFL-based cap
# commented out at this spot, which suggests the same wall was hit; a CFL cap is
# a numerical band-aid that loosens as Δt shrinks, so it is not Δt-convergent.
# The Knudsen term is Δt-independent and physical.
#
# **Why not the MATLAB's D = D_NTP·n_NTP/n_gas.** Written that way v_th cancels
# from the elastic limit entirely, hard-coding 273 K; D then carries no
# temperature dependence at all. Deriving λ from a cross-section keeps D ∝ √T.
# See claudedocs/impurity_model_equations_v2.md.

"H2 molecular mass [kg]."
const M_H2_GAS = 2.01594 * 1.660539e-27

# Kept module-level rather than read from `PlasmaConstants` so the diffusivity
# stays a free function of gas state, testable without a RAPID object. Same value
# as `PlasmaConstants.ee`.
const EE_GAS = 1.602176634e-19

# Reference state for the effective collision cross-section.
#
# SOURCE (⚠ UNVERIFIED — confirm against CRC Handbook / NIST before relying on the
# absolute value): H2 self-diffusion coefficient at 273.15 K, 101325 Pa.
#
# σ is calibrated from a measured TRANSPORT property rather than taken from a
# tabulated molecular diameter, because it is the transport mean free path that
# enters D. The geometric routes disagree badly: the kinetic diameter 2.89 Å and
# the Lennard-Jones σ 2.83 Å both give D ≈ 0.61e-4 m²/s, a factor 2 below the
# measured self-diffusion, i.e. an effective diameter near 1.9 Å.
#
# The factor-2 uncertainty is second order for burn-through: wherever the gas
# actually matters, ν_iz exceeds the elastic rate by 3.5-120×, and where it does
# not, λ already exceeds the vessel so the wall term sets D regardless.
const D_H2_SELF_REF = 1.3e-4                  # [m²/s] @ (T_REF, N_REF)
const N_H2_REF = 2.505e25                     # [m⁻³] Loschmidt @ 273.15 K, 101325 Pa
const T_H2_REF_EV = 273.15 * 8.617333262e-5   # [eV]

"""
    neutral_gas_thermal_speed(T_gas_eV)

Thermal speed `√(T/m)` of the H2 fill [m/s]. This is the `v_th` convention the
diffusivity uses (`D = ½·v_th·λ`), not the mean speed `√(8T/πm)`.
"""
neutral_gas_thermal_speed(T_gas_eV) = sqrt(T_gas_eV * EE_GAS / M_H2_GAS)

# Effective cross-section, inverted from the reference diffusivity under the same
# ½·v_th·λ convention used below, so the model reproduces D_H2_SELF_REF exactly at
# the reference state.
const SIGMA_H2_GAS = let λ_ref = 2 * D_H2_SELF_REF / neutral_gas_thermal_speed(T_H2_REF_EV)
    1 / (sqrt(2) * N_H2_REF * λ_ref)
end

"""
    neutral_gas_diffusivity(n_gas, T_gas_eV, ν_iz, L_char)

Isotropic diffusivity [m²/s] of the neutral H2 fill.

- `n_gas`   molecular density [m⁻³]; may be 0 on burnt-out cells
- `T_gas_eV` gas temperature [eV]
- `ν_iz`    electron-impact ionization frequency seen by a molecule [1/s],
            i.e. `n_e·K_iz` — the rate at which molecules are destroyed
- `L_char`  characteristic vessel dimension [m]; `Inf` disables the wall term

Pass `ν_iz = 0` and `L_char = Inf` to obtain the pure gas-gas value.
"""
function neutral_gas_diffusivity(n_gas, T_gas_eV, ν_iz, L_char)
    v_th = neutral_gas_thermal_speed(T_gas_eV)
    inv_λ = sqrt(2) * n_gas * SIGMA_H2_GAS + ν_iz / v_th + 1 / L_char
    return 0.5 * v_th / inv_λ
end
