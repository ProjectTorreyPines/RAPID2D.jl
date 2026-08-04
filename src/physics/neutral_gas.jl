# Neutral H2 fill: kinetic diffusivity.
#
# A molecule's free path is terminated by whichever happens first — a collision
# with another molecule, destruction by electron impact, or arrival at the wall.
# The three are summed as rates (Matthiessen), so the shortest one dominates:
#
#     1/λ = v_th/(2·D_elastic)  +  ν_iz/v_th  +  1/L
#            ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾      ‾‾‾‾‾‾‾‾‾     ‾‾‾
#            gas-gas (NIST)        ionization    wall (Knudsen)
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

"H2 molecular mass [kg]."
const M_H2_GAS = 2.01594 * 1.660539e-27

# Kept module-level rather than read from `PlasmaConstants` so the diffusivity
# stays a free function of gas state, testable without a RAPID object. Same value
# as `PlasmaConstants.ee`.
const EE_GAS = 1.602176634e-19

const KB_EV_PER_K = 8.617333262e-5      # CODATA Boltzmann constant [eV/K]
const KB_J_PER_K = 1.380649e-23         # CODATA Boltzmann constant [J/K]

# ── H2 self-diffusion: NIST TN 2279 ─────────────────────────────────────────
#
# REFERENCE
#   Burgess DR Jr. (2024). *Self-Diffusion and Binary-Diffusion Coefficients in
#   Gases.* NIST Technical Note (TN) 2279. National Institute of Standards and
#   Technology, Gaithersburg, MD.
#   DOI:  https://doi.org/10.6028/NIST.TN.2279
#   PDF:  https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2279.pdf
#   Data: §3.1.2 Table 1a "Small Molecules", *Hydrogen-Oxygen* block, row 1
#         (p. 8 of the Technical Note).
#
#   Substance  Bath  T_range/K  T_ref/K  D_ref/cm²·s⁻¹     A       B       C   Ref
#   H2         H2    115-295    298      1.309          -9.309  -8.028  1.686   4
#
#   Recommended values are quoted at 101.325 kPa (TN 2279 §1). The temperature
#   dependence is the report's extended form
#       ln(D [cm²/s]) = A + B/T + C·ln(T),   T in K
#   (TN 2279 §1, fit form 1 of 3). Checked: T = 298.15 K reproduces 1.310 cm²/s
#   against the tabulated D_ref = 1.309.
#
# **Self-diffusion (H2 in H2), which is the right quantity for a pure fill.** The
# MATLAB original uses 0.61e-4 m²/s — the H2-in-AIR binary coefficient, a factor
# 2.1 lower. No scaling law would expose that swap, so the test pins the absolute
# value against the number above.
#
# **Why the fit rather than a hard-sphere σ.** C = 1.686 gives D ∝ T^1.69, the
# usual real-gas behaviour once the attractive potential is included; a
# hard-sphere mean free path would give √T and fall ~2.5× low across the fit's
# span. Feeding the measured fit into the elastic channel leaves both limits
# correct on their own terms: elastic-dominated recovers the measured D(T), and
# wall-dominated tends to ½·v_th·L, which scales as √T because free streaming is
# what it describes.
#
# Our T_gas (0.026 eV ≈ 302 K) sits just past the fit's 115-295 K window — the
# same mild extrapolation NIST itself makes in quoting D_298.
const NIST_TN2279_H2_A = -9.309
const NIST_TN2279_H2_B = -8.028
const NIST_TN2279_H2_C = 1.686
"Reference temperature [K] of the TN 2279 recommended value."
const NIST_H2_T_REF_K = 298.15
"Number density [m⁻³] at 298.15 K and 101.325 kPa — the state TN 2279 quotes at."
const NIST_H2_N_REF = 101325.0 / (KB_J_PER_K * NIST_H2_T_REF_K)

"""
    neutral_gas_thermal_speed(T_gas_eV)

Thermal speed `√(T/m)` of the H2 fill [m/s]. This is the `v_th` convention the
diffusivity uses (`D = ½·v_th·λ`), not the mean speed `√(8T/πm)`.
"""
neutral_gas_thermal_speed(T_gas_eV) = sqrt(T_gas_eV * EE_GAS / M_H2_GAS)

"""
    h2_self_diffusivity(T_gas_eV)

H2 self-diffusion coefficient [m²/s] at the reference density `NIST_H2_N_REF`,
evaluated from the NIST TN 2279 Table 1a fit `ln(D) = A + B/T + C·ln(T)`. Being a
dilute-gas transport coefficient it scales as `1/n` away from that density.
"""
function h2_self_diffusivity(T_gas_eV)
    T_K = T_gas_eV / KB_EV_PER_K
    return 1.0e-4 * exp(
        NIST_TN2279_H2_A + NIST_TN2279_H2_B / T_K + NIST_TN2279_H2_C * log(T_K)
    )
end

"""
    neutral_gas_diffusivity(n_gas, T_gas_eV, ν_iz, L_char)

Isotropic diffusivity [m²/s] of the neutral H2 fill.

- `n_gas`    molecular density [m⁻³]; may be 0 on burnt-out cells
- `T_gas_eV` gas temperature [eV]
- `ν_iz`     electron-impact ionization frequency seen by a molecule [1/s],
             i.e. `n_e·K_iz` — the rate at which molecules are destroyed
- `L_char`   characteristic vessel dimension [m]; `Inf` disables the wall term

Pass `ν_iz = 0` and `L_char = Inf` to obtain the pure gas-gas value.
"""
function neutral_gas_diffusivity(n_gas, T_gas_eV, ν_iz, L_char)
    v_th = neutral_gas_thermal_speed(T_gas_eV)
    # measured elastic diffusivity, scaled to the local density, then inverted
    # under this file's D = ½·v_th·λ convention to give a rate
    D_elastic = h2_self_diffusivity(T_gas_eV) * NIST_H2_N_REF / n_gas
    inv_λ = v_th / (2 * D_elastic) + ν_iz / v_th + 1 / L_char
    return 0.5 * v_th / inv_λ
end

"""
    neutral_gas_channel(n_gas, T_gas_eV, ν_iz, L_char)

The same physics as `neutral_gas_diffusivity`, returned as the four numbers of the
transport-channel basis instead of collapsed into one diffusivity.

`neutral_gas_diffusivity` computes `λ` explicitly and then destroys it by
returning `½·v_th·λ`. A diffusivity cannot state a wall condition — `D = ½vλ` is
one equation in two unknowns, and the wall needs the *speed* — so this returns the
pair rather than the product. Nothing new is modelled: `½·v_para·λ_para`
reproduces `neutral_gas_diffusivity` exactly.

**Isotropic**, so `⊥ ≡ ∥`: a neutral molecule has no preferred axis. The three
competing processes — elastic, ionization, wall — all happen at the same `v_th`,
so they combine by Matthiessen *inside* `λ` and their kinetic ceiling is counted
once, not summed.
"""
function neutral_gas_channel(n_gas, T_gas_eV, ν_iz, L_char)
    v_th = neutral_gas_thermal_speed.(T_gas_eV)
    D_elastic = @. h2_self_diffusivity(T_gas_eV) * NIST_H2_N_REF / n_gas
    inv_λ = @. v_th / (2 * D_elastic) + ν_iz / v_th + 1 / L_char
    λ = @. 1 / inv_λ
    return DiffusionChannel(v_th, λ, v_th, λ)
end

"""
    build_reflective_diffusion_matrix(G, D) -> SparseMatrixCSC

Isotropic 5-point diffusion operator `∇·(D∇·)` with a **reflective** (zero-flux)
wall, on the full `NR·NZ` node indexing.

```
    A[p, q] = (1/J_p)·½·(CT_q + CT_p)     q a cardinal neighbour of p, both in-wall
    A[p, p] = −Σ_q A[p, q]
```
with `CT = J·D/dR²` (radial) and `J·D/dZ²` (vertical). Rows for nodes on or
outside the wall are left empty, so those nodes never evolve.

**Reflective, not absorbing.** A neighbour contributes only when it is itself
in-wall; otherwise the term is *omitted*, not zeroed. That distinction is the
whole boundary condition. Zeroing `D` outside instead — the natural thing to try
with the shared `∇𝐃∇` builder, which sweeps every interior node without wall
awareness — still leaves the coefficient `(1/J)·½·CT_inside` on the outward face,
and the gas drains into nodes nothing solves for. The fill gas is not consumed by
the wall; it bounces.

**What is conserved.** Zero row sums make the stencil a divergence, so constants
lie in its kernel and no node manufactures gas. Weighting by the Jacobian makes
`M = J·A` symmetric, hence its column sums vanish too, so `Σ J·n` is conserved
exactly — the same invariant the impurity wall ledger uses. Plain `Σ n` is not
conserved in cylindrical geometry.

Ported from `Construct_An_H2_gas_diffu_reflective.m`. The MATLAB splits the sweep
into deep-in-wall and near-wall passes; that is a performance split only, and the
uniform neighbour test here produces the same matrix.
"""
function build_reflective_diffusion_matrix(G, D::AbstractMatrix{FT}) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    Ng = NR * NZ
    CTRR = @. G.Jacob * D / (G.dR * G.dR)
    CTZZ = @. G.Jacob * D / (G.dZ * G.dZ)
    state, nid = G.nodes.state, G.nodes.nid

    rows = Int[]
    cols = Int[]
    vals = FT[]
    sizehint!(rows, 5 * Ng)
    sizehint!(cols, 5 * Ng)
    sizehint!(vals, 5 * Ng)

    is_inside(i, j) = 1 <= i <= NR && 1 <= j <= NZ && state[i, j] > FT(0.5)

    @inbounds for j in 1:NZ, i in 1:NR
        is_inside(i, j) || continue
        row = nid[i, j]
        invJ = one(FT) / G.Jacob[i, j]
        diag = zero(FT)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            is_inside(ii, jj) || continue          # reflective: omit, do not zero
            CT = dj == 0 ? CTRR : CTZZ
            c = invJ * FT(0.5) * (CT[ii, jj] + CT[i, j])
            push!(rows, row)
            push!(cols, nid[ii, jj])
            push!(vals, c)
            diag -= c
        end
        push!(rows, row)
        push!(cols, row)
        push!(vals, diag)
    end

    return sparse(rows, cols, vals, Ng, Ng)
end

"""
    update_neutral_H2_gas_density!(RP)

Advance the neutral H2 fill one step: burn-out by electron impact, then
reflective diffusion.

```
    ∂n_H2/∂t = ∇·(D∇n_H2) − n_e·ν_iz^{H2}
```

The two halves are operator-split, sink first, matching the MATLAB original.

**The sink is the electron source, not a copy of it.** It reads the very same
`plasma.ν_en_iz` that `solve_electron_continuity_equation!` uses, because one
electron is born for each molecule destroyed. Recomputing the rate here — or
letting a driver script subtract its own estimate — breaks nuclei conservation:
the scenario scripts that did exactly that overshot the electron supply limit by
7% at dt = 1e-5, an error that only vanishes as dt → 0 because the script's sink
was explicit while the electron equation's source was implicit.

Applied on in-wall nodes only, so gas outside the vessel is never consumed.

**Time scheme: `flags.θ_imp.gas`, default 1 (backward Euler).** Deliberately its
own member rather than the transport weight, which is ½ — Crank-Nicolson. CN is
A-stable but not L-stable: its amplification factor tends to −1 as |λ|Δt grows,
so stiff modes ring instead of damping, and D varies by two orders of magnitude
across the shielding layer so the stiff end is always present. The MATLAB
hard-codes the same choice with the note *"safer choice to prevent from
oscillation"*. Exposed as a knob so CN (θ = ½) can be plugged in for a smooth,
well-resolved problem where second-order accuracy outweighs damping; θ = 0 gives
forward Euler and skips the solve, but is bound by the explicit CFL limit
`min(dR,dZ)²/(4D)`.

The diffusivity is evaluated per cell against the *molecular* destruction rate
`n_e·K_iz = n_e·ν_en_iz/n_H2`, not the electron's `ν_en_iz`. Those differ by
`n_e/n_H2` and it is the molecule's fate that sets its free path.
"""
function update_neutral_H2_gas_density!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    @timeit RAPID_TIMER "update_neutral_H2_gas_density!" begin
        pla = RP.plasma
        G = RP.G
        dt = RP.dt
        OP = RP.operators
        zero_FT = zero(FT)

        # ── burn-out ────────────────────────────────────────────────────────
        @inbounds for k in G.nodes.in_wall_nids
            pla.n_H2_gas[k] = max(
                pla.n_H2_gas[k] - dt * pla.ne[k] * pla.ν_en_iz[k], zero_FT
            )
        end

        # ── diffusivity ─────────────────────────────────────────────────────
        # Vessel scale for the Knudsen term: the shorter extent bounds a free path.
        L_char = min(
            maximum(RP.wall.R) - minimum(RP.wall.R),
            maximum(RP.wall.Z) - minimum(RP.wall.Z),
        )
        D = similar(pla.n_H2_gas)
        @inbounds for k in eachindex(D)
            n = pla.n_H2_gas[k]
            # ν seen by a MOLECULE, not by an electron
            ν_iz_gas = n > zero_FT ? pla.ne[k] * pla.ν_en_iz[k] / n : zero_FT
            D[k] = neutral_gas_diffusivity(n, pla.T_gas_eV, ν_iz_gas, L_char)
        end

        # ── reflective diffusion, θ-scheme (BE by default) ──────────────────
        A = build_reflective_diffusion_matrix(G, D)
        θ = RP.flags.θ_imp.gas
        # rows outside the wall are empty in A, so this leaves them as identity
        M = sparse(I, size(A, 1), size(A, 2)) - θ * dt * A
        # The RHS is copied out rather than solved in place. UMFPACK rejects an
        # aliased (X, B) outright for a Vector argument; `view(matrix, :)` builds a
        # different SubArray wrapper that slips past that check while still sharing
        # storage, so an in-place call here would ride on an explicitly forbidden
        # path — correct today only because UMFPACK happens to buffer B internally,
        # and silent if that ever changes or if BandedLUSolver is used instead.
        rhs = vec(pla.n_H2_gas) + (one(FT) - θ) * dt * (A * vec(pla.n_H2_gas))
        @timeit RAPID_TIMER "n_H2_gas LinearSolve" begin
            if θ == zero(FT)
                copyto!(view(pla.n_H2_gas, :), rhs)   # explicit: no solve needed
            else
                factorize!(OP.gas_solver, M)
                solve!(view(pla.n_H2_gas, :), OP.gas_solver, rhs)
            end
        end

        return RP
    end # @timeit
end
