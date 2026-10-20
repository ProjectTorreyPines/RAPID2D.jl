# The current column's zeroth-order force balance: what a reduced position model or an ideal
# vertical-field controller needs from the 2D state. Measured at the 0D snapshots only.

# the names `column_force_balance` returns, in the order the snapshot lists them
const COLUMN_FORCE_BALANCE_FIELDS = (
    :I_closed, :current_coherence, :R_cen_by_Jϕ, :Z_cen_by_Jϕ, :a_eff_by_Jϕ, :Bpol_edge,
    :li, :beta_p, :pressure_column_share,
    :Bv_req, :BZ_presc_at_cen_by_Jϕ, :BR_presc_at_cen_by_Jϕ, :decay_index, :decay_index_radial_limit,
    :force_R_presc, :force_Z_presc, :force_R_coils, :force_Z_coils, :force_R_pla, :force_Z_pla,
    :force_R_pressure, :force_R_toroidal, :Bv_to_add,
    :R_eq, :R_eq_stiffness, :R_axis, :Z_axis,
    :kappa_by_Jϕ, :half_width_minor_by_Jϕ, :half_width_major_by_Jϕ, :force_R_pressure_all, :Br_to_add,
    :force_R_on_closed, :force_Z_on_closed, :Bv_to_add_closed,
    :column_mass, :nu_in_column, :wall_gap_R, :t_wall_inertial, :t_wall_drag,
)

"""
    column_force_balance(RP) -> NamedTuple

The toroidal current column's zeroth-order force balance, a **total-current screening estimate**:
the whole current is taken as one column at its current centroid, with the circular formulas on its
moment ellipse. With `IdealVerticalFieldControl` the forces leave the control field out: they are
the estimate the control cancels, measured on the controlled state, and the control's own values
are the snapshot's `ctrl_*` fields.

- **Column.**
  - `I_closed`: the current on closed field lines.
  - `current_coherence = |∫Jϕ dA| / ∫|Jϕ| dA`: 1 for a one-signed current. Well below 1 the
    centroid and `Bv_to_add` lose their meaning.
  - `R_cen_by_Jϕ`, `Z_cen_by_Jϕ`: the current centroid, |Jϕ|-weighted over the area: for a
    one-signed current, the radius of a filament with the same force in a uniform field. The snapshot's `R_cen_by_Jϕ_Vol` weights the
    signed Jϕ by volume, and `R_cen_by_ne` is the electron-density centroid.
  - `a_eff_by_Jϕ = √(2⟨r²⟩)`, the second moment of the current about that centroid, which gives back
    the radius of a uniform disc; `Bpol_edge = μ0 I / (2π a_eff_by_Jϕ)`, the column's poloidal field
    at that radius (signed with I). On an ellipse of half-widths a, b, a_eff = √((a² + b²)/2) and
    2π a_eff is its perimeter to 1 % at κ = 1.5, so `Bpol_edge` is the perimeter-averaged field.
  - `kappa_by_Jϕ = √(σ_ZZ/σ_RR)`: the elongation of the current from its second moments.
  - **The column's region** is the uniform ellipse with the current's second moments: about the
    centroid, along their principal axes, half-widths 2√λ (`half_width_minor_by_Jϕ`,
    `half_width_major_by_Jϕ`; a uniform ellipse gives itself back, tilted or not; on the grid, the
    nodes inside the ellipse of the discrete moments).
  - `li = 2∫B_p² dV / (μ0² I² R)` inside the region, of the plasma's own field less its mean over
    it: an internal-inductance proxy (2ab/(a + b)² on a uniform ellipse). The removed part is the
    column's own vertical field, which an equilibrium's vertical field cancels up to the pressure
    and toroidal terms below. `beta_p = 8π∫p dA / (μ0 I²)` inside the region, with
    `p = e (nₑ Tₑ + nᵢ Tᵢ)` (2μ0⟨p⟩/B_pa² on a circle). `pressure_column_share`: the part of ∫p dA
    inside it.
- **Shafranov.** `Bv_req = −μ0 I Λ / (4π R)`, `Λ = ln(8R/ā) + β_p + l_i/2 − 3/2`: the vertical
  field the column at the current centroid needs, with ā = (a + b)/2 of its ellipse. At large
  aspect ratio this is the column's self-inductance when the ellipse's edge is a flux surface (an
  elliptic edge's equivalent radius is ā); a uniform-current ellipse, whose edge is not one, adds
  (a − b)²/(4(a + b)²) to Λ. `BZ_presc_at_cen_by_Jϕ` and `BR_presc_at_cen_by_Jϕ`: what the prescribed
  field gives there. This formula side (`Bv_req`, `R_eq`, `decay_index`) uses the prescribed field
  only, not the coils'. `decay_index = −(R/B_Z) ∂B_Z/∂R`
  of the prescribed field there.
  At fixed current a circular column is stable for `0 < decay_index < 1 − 1/Λ`
  (`decay_index_radial_limit`); the conventional screen is `0 < n < 3/2`.
- **Forces [N]**, with `dV = 2πR dR dZ`:
  - `force_R_presc = ∫ Jϕ B_Z,presc dV`, `force_Z_presc = −∫ Jϕ B_R,presc dV`;
  - `force_R_coils`, `force_Z_coils`: the same with the field of the coils the grid solves;
  - `force_R_pla`, `force_Z_pla`: the same with the plasma's own field (the hoop force);
  - `force_R_pressure = −∫ ∂p/∂R dV` over the column's region: the tire tube 2π ∫ p dR dZ less
    the pressure outside on its edge, 2π ∮ p R n_R dZ (zero when the pressure is uniform across the
    edge, the tire tube when it ends inside);
  - `force_R_toroidal = (μ0 I²/4)(β_p − 1)`, the toroidal-field force of a circular column in
    minor-radius equilibrium: a closure, since the model has no δB_ϕ.
  - `force_R_pressure_all = 2π ∫ p dR dZ` of all the pressure on the grid. Against
    `force_R_pressure` it shows how much the choice of the column's support matters.
- **Missing fields.** `Bv_to_add = −F_R / ∫ Jϕ dV`, with `F_R` the sum of the five radial forces:
  the uniform B_Z that, added, cancels this reduced radial force estimate. It includes the
  toroidal closure. `Bv_req − BZ_presc_at_cen_by_Jϕ` estimates the same from the formula (without
  the coils' field). `Br_to_add = F_Z / ∫ Jϕ dV`: the uniform B_R that cancels the net vertical
  force. Both NaN when the signed ∫ Jϕ dV cancels.
- **The current on closed field lines**, by the same reduced estimate on that region, as an
  isolated column: `force_R_on_closed`, `force_Z_on_closed` (the forces of all the fields on its
  current, the pressure force on that region, and the toroidal closure with its own current,
  centroid and β_p), and `Bv_to_add_closed`, the uniform B_Z that cancels that radial force. The
  stresses of the open current around it are not counted.
- **If the column could move** (limits, not dynamics): `column_mass`, the ion mass inside the
  column's region; `nu_in_column`, its ion–neutral drag rate; `wall_gap_R`, how far the centroid
  would travel along its grid row to the wall in the direction of the net radial force, inside
  the run of in-wall nodes that holds it; and the time to close that gap at constant force,
  `t_wall_inertial = √(2 M gap / |F_R|)` and the drag-limited `t_wall_drag = gap M ν / |F_R|`
  (Inf at zero force). A large `ν t_wall_inertial` means the drag limit applies. The force is the
  whole current's and the mass the region's: screening limits, not an acceleration to apply.
- **Positions.** `R_eq`: on the centroid's midplane, the root of
  `F_R(R) = (μ0 I²/2) Λ(R) + 2πR I B_Z,presc(R)` nearest the centroid, with `a`, `β_p`, `l_i` held,
  among its sign changes sampled across the grid (sampled zeros count; tangent roots are missed),
  each refined by bisection; `R_eq_stiffness = dF_R/dR` there (negative is restoring). `R_axis`,
  `Z_axis`: the magnetic axis, the constraint basis's when there is one (not rechecked against the
  present field), else the single converged O-point of the present ψ in the stored closed region.

Values that do not exist are `NaN`: all of them below `flags.Ampere_Itor_threshold` of current;
the fields to add where ∫ Jϕ dV cancels; `decay_index` where B_Z,presc vanishes; `R_eq` and its
stiffness without a detected root; the axis without a single converged O-point in the closed
region; the wall gap and times when the centroid is not inside the wall on its row; and everything
built on the plasma's or the coils' field when the run does not solve them (Ampère off).
"""
function column_force_balance(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, F, pla = RP.G, RP.fields, RP.plasma
    μ0 = RP.config.constants.μ0
    J = pla.Jϕ
    dA = G.dR * G.dZ
    I = sum(J) * dA
    if iszero(I) || abs(I) < RP.flags.Ampere_Itor_threshold
        return NamedTuple{COLUMN_FORCE_BALANCE_FIELDS}(ntuple(_ -> FT(NaN), length(COLUMN_FORCE_BALANCE_FIELDS)))
    end
    V = G.inVol2D

    closed = [k for k in RP.flf.closed_surface_nids if G.nodes.state[k] > 0.5]
    I_closed = sum(J[k] for k in closed; init = zero(FT)) * dA

    # the plasma's own field and the coils', each from its own currents
    BR_pla, BZ_pla = _plasma_poloidal_field(RP)
    BR_coils, BZ_coils = _coil_poloidal_field(RP)

    # the column (centroid, moments, ellipse, β_p) and the forces on it, shared with the
    # vertical-field control so that both cancel and report the same estimate
    col = _column_forces(RP, BR_pla, BZ_pla, BR_coils, BZ_coils)
    (; current_coherence, R_cen_by_Jϕ, Z_cen_by_Jϕ, a_eff_by_Jϕ, kappa_by_Jϕ, inside) = col
    (; half_width_minor_by_Jϕ, half_width_major_by_Jϕ, p, p_area, beta_p) = col
    (; force_R_presc, force_Z_presc, force_R_coils, force_Z_coils, force_R_pla, force_Z_pla) = col
    (; force_R_pressure, force_R_toroidal, force_R_net, force_Z_net, ∫J, net_current) = col
    Bpol_edge = μ0 * I / (2π * a_eff_by_Jϕ)
    V_in = V[inside]

    # l_i from the plasma's field less its mean over the column: that uniform part is the column's
    # own vertical field (its hoop force), which the vertical field of an equilibrium cancels
    BR_in, BZ_in = BR_pla[inside], BZ_pla[inside]
    BR_in .-= sum(BR_in .* V_in) / sum(V_in)
    BZ_in .-= sum(BZ_in .* V_in) / sum(V_in)
    li = 2 * sum(@. (BR_in^2 + BZ_in^2) * V_in) / (μ0^2 * I^2 * R_cen_by_Jϕ)
    pressure_column_share = sum(p_area) > 0 ? sum(p_area[inside]) / sum(p_area) : FT(NaN)

    # Shafranov: the vertical field the column needs, and the prescribed field it has
    Λ(R) = log(8R / ((half_width_minor_by_Jϕ + half_width_major_by_Jϕ) / 2)) + beta_p + li / 2 - FT(1.5)
    Bv_req = -μ0 * I * Λ(R_cen_by_Jϕ) / (4π * R_cen_by_Jϕ)
    BZ_presc_at_cen_by_Jϕ = _bilinear(G, F.BZ_presc, R_cen_by_Jϕ, Z_cen_by_Jϕ)
    BR_presc_at_cen_by_Jϕ = _bilinear(G, F.BR_presc, R_cen_by_Jϕ, Z_cen_by_Jϕ)
    h = G.dR / 2
    dBZ_dR = (_bilinear(G, F.BZ_presc, R_cen_by_Jϕ + h, Z_cen_by_Jϕ) - _bilinear(G, F.BZ_presc, R_cen_by_Jϕ - h, Z_cen_by_Jϕ)) / (2h)
    decay_index = iszero(BZ_presc_at_cen_by_Jϕ) ? FT(NaN) : -R_cen_by_Jϕ / BZ_presc_at_cen_by_Jϕ * dBZ_dR
    decay_index_radial_limit = one(FT) - one(FT) / Λ(R_cen_by_Jϕ)

    # the uniform fields that cancel the net radial and vertical forces (F_Z = −∫Jϕ B_R dV)
    Bv_to_add = net_current ? -force_R_net / ∫J : FT(NaN)
    Br_to_add = net_current ? force_Z_net / ∫J : FT(NaN)

    # the tire tube of all the pressure on the grid, against the column's: how much the support matters
    force_R_pressure_all = sum(p_area)

    # the current on closed field lines, by the same reduced estimate on that region as an isolated
    # column: the forces of all the fields on its current, the pressure force on the region, and the
    # toroidal closure with its own current, centroid and β_p
    force_R_on_closed, force_Z_on_closed, Bv_to_add_closed = FT(NaN), FT(NaN), FT(NaN)
    if !isempty(closed) && !iszero(I_closed)
        BZ_all, BR_all = F.BZ_presc .+ BZ_coils .+ BZ_pla, F.BR_presc .+ BR_coils .+ BR_pla
        R_cen_closed = sum(abs(J[k]) * G.R2D[k] for k in closed) / sum(abs(J[k]) for k in closed)
        beta_p_closed = 4 * sum(p[k] * V[k] for k in closed) / (μ0 * I_closed^2 * R_cen_closed)
        on_closed = falses(size(J))
        on_closed[closed] .= true
        force_R_on_closed = sum(J[k] * BZ_all[k] * V[k] for k in closed) + _pressure_force_R(G, p, V, on_closed) +
            μ0 * I_closed^2 / 4 * (beta_p_closed - 1)
        force_Z_on_closed = -sum(J[k] * BR_all[k] * V[k] for k in closed)
        ∫J_closed = sum(J[k] * V[k] for k in closed)
        if _uncancelled(∫J_closed, sum(abs(J[k]) * V[k] for k in closed))
            Bv_to_add_closed = -force_R_on_closed / ∫J_closed
        end
    end

    # if the column could move: its ion mass, the ion–neutral drag rate, the gap to the wall on the
    # centroid's midplane in the direction of the net radial force, and the time to close it at
    # constant force, inertial (√(2Mℓ/|F|)) and drag-limited (ℓ M ν / |F|). Limits, not dynamics
    ni_V_in = pla.ni[inside] .* V_in
    column_mass = bulk_ion_mass(RP) * sum(ni_V_in)
    nu_in_column = sum(ni_V_in) > 0 ? sum(RP.transport.νi_neutral[inside] .* ni_V_in) / sum(ni_V_in) : FT(NaN)
    wall_gap_R = _wall_gap_R(G, R_cen_by_Jϕ, Z_cen_by_Jϕ, force_R_net)
    t_wall_inertial = sqrt(2 * column_mass * wall_gap_R / abs(force_R_net))
    t_wall_drag = wall_gap_R * column_mass * nu_in_column / abs(force_R_net)

    # the equilibrium radius of the formula on the centroid's midplane, nearest the centroid
    force_R_formula(R) = μ0 * I^2 / 2 * Λ(R) + 2π * R * I * _bilinear(G, F.BZ_presc, R, Z_cen_by_Jϕ)
    R_eq = _root_nearest(force_R_formula, G.R1D[2], G.R1D[end - 1], 4 * G.NR, R_cen_by_Jϕ)
    R_eq_stiffness = isnan(R_eq) ? FT(NaN) : (force_R_formula(R_eq + h) - force_R_formula(R_eq - h)) / (2h)

    R_axis, Z_axis = _magnetic_axis(RP, closed)

    return (;
        I_closed, current_coherence, R_cen_by_Jϕ, Z_cen_by_Jϕ, a_eff_by_Jϕ, Bpol_edge,
        li, beta_p, pressure_column_share,
        Bv_req, BZ_presc_at_cen_by_Jϕ, BR_presc_at_cen_by_Jϕ, decay_index, decay_index_radial_limit,
        force_R_presc, force_Z_presc, force_R_coils, force_Z_coils, force_R_pla, force_Z_pla,
        force_R_pressure, force_R_toroidal, Bv_to_add,
        R_eq, R_eq_stiffness, R_axis, Z_axis,
        kappa_by_Jϕ, half_width_minor_by_Jϕ, half_width_major_by_Jϕ, force_R_pressure_all, Br_to_add,
        force_R_on_closed, force_Z_on_closed, Bv_to_add_closed,
        column_mass, nu_in_column, wall_gap_R, t_wall_inertial, t_wall_drag,
    )
end

# The column and the reduced forces on it, from the fields of the present currents (`BR_pla`,
# `BZ_pla`, `BR_coils`, `BZ_coils`) and the prescribed field; the control field is left out. The
# centroid, second moments and moment ellipse of |Jϕ|, the pressure and β_p in the ellipse, the
# forces of each field on Jϕ, the pressure force on the ellipse and the toroidal closure, and
# their radial and vertical sums. Call it above the current threshold (I ≠ 0)
function _column_forces(RP::RAPID{FT}, BR_pla, BZ_pla, BR_coils, BZ_coils) where {FT <: AbstractFloat}
    G, F, pla = RP.G, RP.fields, RP.plasma
    μ0, ee = RP.config.constants.μ0, RP.config.constants.ee
    J, V = pla.Jϕ, G.inVol2D
    dA = G.dR * G.dZ
    I = sum(J) * dA

    weight = abs.(J)
    current_coherence = abs(sum(J)) / sum(weight)
    R_cen_by_Jϕ = sum(weight .* G.R2D) / sum(weight)
    Z_cen_by_Jϕ = sum(weight .* G.Z2D) / sum(weight)
    x, z = G.R2D .- R_cen_by_Jϕ, G.Z2D .- Z_cen_by_Jϕ
    # a uniform ellipse of half-widths a, b along R, Z has σ_RR = a²/4, σ_ZZ = b²/4
    σ_RR, σ_RZ, σ_ZZ = (sum(weight .* q) / sum(weight) for q in (x .^ 2, x .* z, z .^ 2))
    one_cell = sqrt(dA / π)                                   # the radius of one cell's disc
    a_eff_by_Jϕ = max(sqrt(2 * (σ_RR + σ_ZZ)), one_cell)
    kappa_by_Jϕ = σ_RR > 0 ? sqrt(σ_ZZ / σ_RR) : FT(NaN)
    inside, half_width_minor_by_Jϕ, half_width_major_by_Jϕ = _moment_ellipse(x, z, σ_RR, σ_RZ, σ_ZZ, one_cell)

    p = @. ee * (pla.ne * pla.Te_eV + pla.ni * pla.Ti_eV)
    beta_p = 4 * sum(p[inside] .* V[inside]) / (μ0 * I^2 * R_cen_by_Jϕ)
    p_area = p .* V ./ G.R2D                                  # 2π p dA

    force_R_presc = sum(J .* F.BZ_presc .* V)
    force_Z_presc = -sum(J .* F.BR_presc .* V)
    force_R_coils = sum(J .* BZ_coils .* V)
    force_Z_coils = -sum(J .* BR_coils .* V)
    force_R_pla = sum(J .* BZ_pla .* V)
    force_Z_pla = -sum(J .* BR_pla .* V)
    force_R_pressure = _pressure_force_R(G, p, V, inside)
    force_R_toroidal = μ0 * I^2 / 4 * (beta_p - 1)
    force_R_net = force_R_presc + force_R_coils + force_R_pla + force_R_pressure + force_R_toroidal
    force_Z_net = force_Z_presc + force_Z_coils + force_Z_pla
    ∫J = sum(J .* V)
    net_current = _uncancelled(∫J, sum(abs.(J) .* V))         # a uniform field acts on the net current
    return (;
        I, current_coherence, R_cen_by_Jϕ, Z_cen_by_Jϕ, a_eff_by_Jϕ, kappa_by_Jϕ, inside,
        half_width_minor_by_Jϕ, half_width_major_by_Jϕ, p, p_area, beta_p,
        force_R_presc, force_Z_presc, force_R_coils, force_Z_coils, force_R_pla, force_Z_pla,
        force_R_pressure, force_R_toroidal, force_R_net, force_Z_net, ∫J, net_current,
    )
end

# The uniform ellipse with second moments σ about the centroid (offsets x, z): along the principal
# axes, half-widths 2√λ for the eigenvalues λ, each at least `min_half_width`. Returns the nodes
# inside it and its minor and major half-widths
function _moment_ellipse(x, z, σ_RR::FT, σ_RZ::FT, σ_ZZ::FT, min_half_width::FT) where {FT <: AbstractFloat}
    mean, spread = (σ_RR + σ_ZZ) / 2, hypot((σ_RR - σ_ZZ) / 2, σ_RZ)
    tilt = atan(2σ_RZ, σ_RR - σ_ZZ) / 2                       # the major axis, from the R direction
    half_major = max(2 * sqrt(mean + spread), min_half_width)
    half_minor = max(2 * sqrt(max(mean - spread, zero(FT))), min_half_width)
    c, s = cos(tilt), sin(tilt)
    inside = @. ((x * c + z * s) / half_major)^2 + ((z * c - x * s) / half_minor)^2 <= 1
    return inside, half_minor, half_major
end

# On the grid row nearest Z, the distance from R to the end of the run of in-wall nodes that holds
# R, in the direction of the radial force F (outboard for F > 0, inboard for F < 0, the nearer end
# at F = 0). NaN when the node nearest (R, Z) is not in the wall
function _wall_gap_R(G::GridGeometry{FT}, R::FT, Z::FT, F::FT) where {FT <: AbstractFloat}
    (isfinite(R) && isfinite(Z)) || return FT(NaN)
    j = clamp(round(Int, (Z - G.Z1D[1]) / G.dZ) + 1, 1, G.NZ)
    i = clamp(round(Int, (R - G.R1D[1]) / G.dR) + 1, 1, G.NR)
    in_wall(k) = G.nodes.state[k, j] > 0.5
    in_wall(i) || return FT(NaN)
    i_in, i_out = i, i                                        # the run's inboard and outboard ends
    while i_in > 1 && in_wall(i_in - 1)
        i_in -= 1
    end
    while i_out < G.NR && in_wall(i_out + 1)
        i_out += 1
    end
    inboard, outboard = max(R - G.R1D[i_in], zero(FT)), max(G.R1D[i_out] - R, zero(FT))
    F > 0 && return outboard
    F < 0 && return inboard
    return min(inboard, outboard)
end

# Whether a signed sum survives its cancellation, relative to the sum of its magnitudes
_uncancelled(signed::FT, magnitude::FT) where {FT <: AbstractFloat} = abs(signed) > sqrt(eps(FT)) * magnitude

# The radial force of the pressure on a region of the grid, −∫ ∂p/∂R dV: the tire tube 2π ∫ p dA
# over it less 2π ∮ p R n_R dZ on its R faces, with p from the node outside (zero beyond the grid).
# Uniform pressure across the edge gives zero; pressure that ends inside gives the tire tube
function _pressure_force_R(G::GridGeometry{FT}, p::AbstractMatrix{FT}, V::AbstractMatrix{FT}, region::AbstractMatrix{Bool}) where {FT <: AbstractFloat}
    force = zero(FT)
    for j in 1:G.NZ, i in 1:G.NR
        region[i, j] || continue
        force += p[i, j] * V[i, j] / G.R2D[i, j]
        if i < G.NR && !region[i + 1, j]
            force -= 2π * p[i + 1, j] * (G.R1D[i] + G.dR / 2) * G.dZ
        end
        if i > 1 && !region[i - 1, j]
            force += 2π * p[i - 1, j] * (G.R1D[i] - G.dR / 2) * G.dZ
        end
    end
    return force
end

# The root of f in [lo, hi] nearest x0 among its sign changes on n samples (a sampled zero counts),
# each refined by bisection; NaN without one. Tangent roots, and two roots in one interval, are missed
function _root_nearest(f, lo::FT, hi::FT, n::Integer, x0::FT) where {FT <: AbstractFloat}
    xs = range(lo, hi; length = n)
    fs = f.(xs)
    best = FT(NaN)
    for k in 1:n
        if iszero(fs[k])
            root = FT(xs[k])
        elseif k < n && fs[k] * fs[k + 1] < 0                    # a sign change (NaN never is)
            a, b, fa = FT(xs[k]), FT(xs[k + 1]), fs[k]
            for _ in 1:40
                m = (a + b) / 2
                fm = f(m)
                if signbit(fm) == signbit(fa)
                    a, fa = m, fm
                else
                    b = m
                end
            end
            root = (a + b) / 2
        else
            continue
        end
        (isnan(best) || abs(root - x0) < abs(best - x0)) && (best = root)
    end
    return best
end

# whether the run solves the coils' and the plasma's field: the boundary Green function and the
# Grad–Shafranov operator are built only with Ampère on
_solves_coils_pla_field(RP::RAPID) = !isempty(RP.G.Green_inWall2bdy)

# The plasma's own poloidal field, from its stored Jϕ (`plasma_flux_on_grid`, one back-substitution).
# NaN fields when the run does not solve it.
function _plasma_poloidal_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G = RP.G
    BR, BZ = zeros(FT, G.NR, G.NZ), zeros(FT, G.NR, G.NZ)
    if !_solves_coils_pla_field(RP)
        fill!(BR, FT(NaN))
        fill!(BZ, FT(NaN))
        return BR, BZ
    end
    calculate_B_from_ψ!(G, plasma_flux_on_grid(RP, RP.plasma.Jϕ), BR, BZ)
    return BR, BZ
end

# the poloidal field of the coils the grid solves, at their present currents: zero without coils,
# NaN when the run does not solve it
function _coil_poloidal_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, csys = RP.G, RP.coil_system
    BR, BZ = zeros(FT, G.NR, G.NZ), zeros(FT, G.NR, G.NZ)
    csys.n_total > 0 || return BR, BZ
    if !_solves_coils_pla_field(RP)
        fill!(BR, FT(NaN))
        fill!(BZ, FT(NaN))
        return BR, BZ
    end
    calculate_B_from_ψ!(G, coil_flux_on_grid(RP, get_all_currents(csys)), BR, BZ)
    return BR, BZ
end

# The magnetic axis: the constraint basis's when there is one, else the O-point of the closed
# region (`closed`, in-wall linear ids) on the total ψ. NaN without a single converged O-point.
function _magnetic_axis(RP::RAPID{FT}, closed::AbstractVector{<:Integer}) where {FT <: AbstractFloat}
    basis = RP.closed_surface.basis
    basis === nothing || return FT(basis.axis.R), FT(basis.axis.Z)
    isempty(closed) && return FT(NaN), FT(NaN)
    ψ = RP.fields.ψ
    itp = psi_interpolant(RP.G, ψ)
    o = find_o_point(RP.G, ψ, closed; itp)
    (o === nothing || !o.converged || count_o_points(RP.G, ψ, closed, itp) > 1) && return FT(NaN), FT(NaN)
    return FT(o.R), FT(o.Z)
end

# the bilinear value of a grid field at (R, Z), held at the edge value outside the grid
function _bilinear(G::GridGeometry{FT}, field::AbstractMatrix{FT}, R::Real, Z::Real) where {FT <: AbstractFloat}
    x = (R - G.R1D[1]) / G.dR + 1
    y = (Z - G.Z1D[1]) / G.dZ + 1
    i = clamp(floor(Int, x), 1, G.NR - 1)
    j = clamp(floor(Int, y), 1, G.NZ - 1)
    s, t = clamp(x - i, 0, 1), clamp(y - j, 0, 1)
    return FT((1 - s) * (1 - t) * field[i, j] + s * (1 - t) * field[i + 1, j] + (1 - s) * t * field[i, j + 1] + s * t * field[i + 1, j + 1])
end
