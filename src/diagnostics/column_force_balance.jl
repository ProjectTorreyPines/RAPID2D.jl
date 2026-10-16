# The current column's zeroth-order force balance: what a reduced position model or an ideal
# vertical-field controller needs from the 2D state. Measured at the 0D snapshots only.

# the names `column_force_balance` returns, in the order the snapshot lists them
const COLUMN_FORCE_BALANCE_FIELDS = (
    :I_closed, :current_coherence, :R_cen_by_Jϕ, :Z_cen_by_Jϕ, :a_eff_by_Jϕ, :Bpol_edge,
    :li, :beta_p, :pressure_column_share,
    :Bv_req, :BZ_ext_at_cen_by_Jϕ, :BR_ext_at_cen_by_Jϕ, :decay_index, :decay_index_radial_limit,
    :force_R_ext, :force_Z_ext, :force_R_coils, :force_Z_coils, :force_R_self, :force_Z_self,
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
moment ellipse.

- **Column.**
  - `I_closed`: the current on closed field lines.
  - `current_coherence = |∫Jϕ dA| / ∫|Jϕ| dA`: 1 for a one-signed current. Well below 1 the
    centroid and `Bv_to_add` lose their meaning.
  - `R_cen_by_Jϕ`, `Z_cen_by_Jϕ`: the current centroid, |Jϕ|-weighted over the area: the radius of
    a filament with the same force in a uniform field. The snapshot's `R_cen_by_Jϕ_Vol` weights the
    signed Jϕ by volume, and `R_cen_by_ne` is the electron-density centroid.
  - `a_eff_by_Jϕ = √(2⟨r²⟩)`, the second moment of the current about that centroid, which gives back
    the radius of a uniform disc; `Bpol_edge = μ0 I / (2π a_eff_by_Jϕ)`, the column's poloidal field
    at that radius (signed with I). On an ellipse of half-widths a, b, a_eff = √((a² + b²)/2) and
    2π a_eff is its perimeter to 1 % at κ = 1.5, so `Bpol_edge` is the perimeter-averaged field.
  - `kappa_by_Jϕ = √(σ_ZZ/σ_RR)`: the elongation of the current from its second moments.
  - **The column's region** is the uniform ellipse with the current's second moments: about the
    centroid, along their principal axes, half-widths 2√λ (`half_width_minor_by_Jϕ`,
    `half_width_major_by_Jϕ`; a uniform ellipse gives itself back, tilted or not). `li` and
    `beta_p` inside it, from `p = e (nₑ Tₑ + nᵢ Tᵢ)` and the plasma's own
    field less its mean over it (the column's own vertical field, which an equilibrium cancels).
    `pressure_column_share`: the part of ∫p dA inside it.
- **Shafranov.** `Bv_req = −μ0 I Λ / (4π R)`, `Λ = ln(8R/ā) + β_p + l_i/2 − 3/2`: the vertical
  field the column at the current centroid needs, with ā = (a + b)/2 of its ellipse (an ellipse's
  geometric mean distance is ā e^(−1/4), Maxwell). On a uniform κ = 2 ellipse it stays within 1 %
  of the force estimate `Bv_to_add`; with a_eff 3 %, with √(ab) 1.4 %. `BZ_ext_at_cen_by_Jϕ` and
  `BR_ext_at_cen_by_Jϕ`: what the external field gives there. `decay_index = −(R/B_Z) ∂B_Z/∂R`
  of the external field there.
  At fixed current a circular column is stable for `0 < decay_index < 1 − 1/Λ`
  (`decay_index_radial_limit`); the conventional screen is `0 < n < 3/2`.
- **Forces [N]**, with `dV = 2πR dR dZ`:
  - `force_R_ext = ∫ Jϕ B_Z,ext dV`, `force_Z_ext = −∫ Jϕ B_R,ext dV`;
  - `force_R_coils`, `force_Z_coils`: the same with the field of the coils the grid solves;
  - `force_R_self`, `force_Z_self`: the same with the plasma's own field (the hoop force);
  - `force_R_pressure = 2π ∫ p dR dZ` over the column's region, the tire-tube force;
  - `force_R_toroidal = (μ0 I²/4)(β_p − 1)`, the toroidal-field force of a circular column in
    minor-radius equilibrium: a closure, since the model has no δB_ϕ.
  - `force_R_pressure_all`: the tire tube of all the pressure on the grid. Against
    `force_R_pressure` it shows how much the choice of the column's support matters.
- **Missing fields.** `Bv_to_add = −F_R / ∫ Jϕ dV`, with `F_R` the sum of the five radial forces:
  the uniform B_Z that, added, cancels this reduced radial force estimate. It includes the
  toroidal closure. `Bv_req − BZ_ext_at_cen_by_Jϕ` estimates the same from the formula.
  `Br_to_add = F_Z / ∫ Jϕ dV`: the uniform B_R that cancels the net vertical force.
- **The current on closed field lines**, by the same reduced estimate on that region:
  `force_R_on_closed`, `force_Z_on_closed` (the forces of all the fields on its current, the tire
  tube of its pressure, and the toroidal closure with its own current and β_p), and
  `Bv_to_add_closed`, the uniform B_Z that cancels that radial force.
- **If the column could move** (limits, not dynamics): `column_mass`, the ion mass inside the
  column's region; `nu_in_column`, its ion–neutral drag rate; `wall_gap_R`, the distance from the centroid
  to the wall on its midplane in the direction of the net radial force; and the time to close
  that gap at constant force, `t_wall_inertial = √(2 M gap / |F_R|)` and the drag-limited
  `t_wall_drag = gap M ν / |F_R|`. A large `ν t_wall_inertial` means the drag limit applies.
- **Positions.** `R_eq`: on the centroid's midplane, the radius nearest the centroid where
  `F_R(R) = (μ0 I²/2) Λ(R) + 2πR I B_Z,ext(R)` vanishes, with `a`, `β_p`, `l_i` held, and
  `R_eq_stiffness = dF_R/dR` there (negative is restoring). `R_axis`, `Z_axis`: the magnetic
  axis, the O-point of the closed region.

Values that do not exist are `NaN`: all of them below `flags.Ampere_Itor_threshold` of current;
`decay_index` where B_Z,ext vanishes; `R_eq` and its stiffness where `F_R(R)` has no root; the
axis without a single converged O-point in the closed region; and everything built on the
plasma's or the coils' field when the run solves no self field (Ampère off).
"""
function column_force_balance(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, F, pla = RP.G, RP.fields, RP.plasma
    μ0, ee = RP.config.constants.μ0, RP.config.constants.ee
    J = pla.Jϕ
    dA = G.dR * G.dZ
    I = sum(J) * dA
    if iszero(I) || abs(I) < RP.flags.Ampere_Itor_threshold
        return NamedTuple{COLUMN_FORCE_BALANCE_FIELDS}(ntuple(_ -> FT(NaN), length(COLUMN_FORCE_BALANCE_FIELDS)))
    end
    V = G.inVol2D

    # the column: centroid, second moments, and l_i, β_p inside its ellipse
    closed = [k for k in RP.flf.closed_surface_nids if G.nodes.state[k] > 0.5]
    I_closed = sum(J[k] for k in closed; init = zero(FT)) * dA
    weight = abs.(J)
    current_coherence = abs(sum(J)) / sum(weight)
    R_cen_by_Jϕ = sum(weight .* G.R2D) / sum(weight)
    Z_cen_by_Jϕ = sum(weight .* G.Z2D) / sum(weight)
    x, z = G.R2D .- R_cen_by_Jϕ, G.Z2D .- Z_cen_by_Jϕ
    # a uniform ellipse of half-widths a, b along R, Z has σ_RR = a²/4, σ_ZZ = b²/4
    σ_RR, σ_RZ, σ_ZZ = (sum(weight .* q) / sum(weight) for q in (x .^ 2, x .* z, z .^ 2))
    one_cell = sqrt(dA / π)                                   # the radius of one cell's disc
    a_eff_by_Jϕ = max(sqrt(2 * (σ_RR + σ_ZZ)), one_cell)
    Bpol_edge = μ0 * I / (2π * a_eff_by_Jϕ)
    kappa_by_Jϕ = σ_RR > 0 ? sqrt(σ_ZZ / σ_RR) : FT(NaN)
    inside, half_width_minor_by_Jϕ, half_width_major_by_Jϕ = _moment_ellipse(x, z, σ_RR, σ_RZ, σ_ZZ, one_cell)
    V_in = V[inside]

    # the plasma's own field and the coils', each from its own currents
    BR_pla, BZ_pla = _plasma_poloidal_field(RP)
    BR_coils, BZ_coils = _coil_poloidal_field(RP)

    # l_i from the plasma's field less its mean over the column: that uniform part is the column's
    # own vertical field (its hoop force), which the vertical field of an equilibrium cancels
    BR_in, BZ_in = BR_pla[inside], BZ_pla[inside]
    BR_in .-= sum(BR_in .* V_in) / sum(V_in)
    BZ_in .-= sum(BZ_in .* V_in) / sum(V_in)
    li = 2 * sum(@. (BR_in^2 + BZ_in^2) * V_in) / (μ0^2 * I^2 * R_cen_by_Jϕ)
    p = @. ee * (pla.ne * pla.Te_eV + pla.ni * pla.Ti_eV)
    beta_p = 4 * sum(p[inside] .* V_in) / (μ0 * I^2 * R_cen_by_Jϕ)
    p_area = p .* V ./ G.R2D                                  # 2π p dA
    pressure_column_share = sum(p_area) > 0 ? sum(p_area[inside]) / sum(p_area) : FT(NaN)

    # Shafranov: the vertical field the column needs, and the external field it has
    Λ(R) = log(8R / ((half_width_minor_by_Jϕ + half_width_major_by_Jϕ) / 2)) + beta_p + li / 2 - FT(1.5)
    Bv_req = -μ0 * I * Λ(R_cen_by_Jϕ) / (4π * R_cen_by_Jϕ)
    BZ_ext_at_cen_by_Jϕ = _bilinear(G, F.BZ_ext, R_cen_by_Jϕ, Z_cen_by_Jϕ)
    BR_ext_at_cen_by_Jϕ = _bilinear(G, F.BR_ext, R_cen_by_Jϕ, Z_cen_by_Jϕ)
    h = G.dR / 2
    dBZ_dR = (_bilinear(G, F.BZ_ext, R_cen_by_Jϕ + h, Z_cen_by_Jϕ) - _bilinear(G, F.BZ_ext, R_cen_by_Jϕ - h, Z_cen_by_Jϕ)) / (2h)
    decay_index = iszero(BZ_ext_at_cen_by_Jϕ) ? FT(NaN) : -R_cen_by_Jϕ / BZ_ext_at_cen_by_Jϕ * dBZ_dR
    decay_index_radial_limit = one(FT) - one(FT) / Λ(R_cen_by_Jϕ)

    # the forces: of the fields, the tire tube over the column, and the toroidal closure
    force_R_ext = sum(J .* F.BZ_ext .* V)
    force_Z_ext = -sum(J .* F.BR_ext .* V)
    force_R_coils = sum(J .* BZ_coils .* V)
    force_Z_coils = -sum(J .* BR_coils .* V)
    force_R_self = sum(J .* BZ_pla .* V)
    force_Z_self = -sum(J .* BR_pla .* V)
    force_R_pressure = sum(p_area[inside])
    force_R_toroidal = μ0 * I^2 / 4 * (beta_p - 1)
    force_R_net = force_R_ext + force_R_coils + force_R_self + force_R_pressure + force_R_toroidal
    ∫J = sum(J .* V)
    Bv_to_add = -force_R_net / ∫J
    force_Z_net = force_Z_ext + force_Z_coils + force_Z_self
    Br_to_add = force_Z_net / ∫J                              # the uniform B_R that cancels F_Z = −∫Jϕ B_R dV

    # the tire tube of all the pressure on the grid, against the column's: how much the support matters
    force_R_pressure_all = sum(p_area)

    # the current on closed field lines, by the same reduced estimate on that region: the forces of
    # all the fields on its current, the tire tube of its pressure, and the toroidal closure with its
    # own current and β_p
    force_R_on_closed, force_Z_on_closed, Bv_to_add_closed = FT(NaN), FT(NaN), FT(NaN)
    if !isempty(closed) && !iszero(I_closed)
        BZ_all, BR_all = F.BZ_ext .+ BZ_coils .+ BZ_pla, F.BR_ext .+ BR_coils .+ BR_pla
        beta_p_closed = 4 * sum(p[k] * V[k] for k in closed) / (μ0 * I_closed^2 * R_cen_by_Jϕ)
        force_R_on_closed = sum(J[k] * BZ_all[k] * V[k] + p_area[k] for k in closed) +
            μ0 * I_closed^2 / 4 * (beta_p_closed - 1)
        force_Z_on_closed = -sum(J[k] * BR_all[k] * V[k] for k in closed)
        Bv_to_add_closed = -force_R_on_closed / sum(J[k] * V[k] for k in closed)
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
    force_R_formula(R) = μ0 * I^2 / 2 * Λ(R) + 2π * R * I * _bilinear(G, F.BZ_ext, R, Z_cen_by_Jϕ)
    R_scan = range(G.R1D[2], G.R1D[end - 1]; length = 4 * G.NR)
    F_scan = force_R_formula.(R_scan)
    R_eq, R_eq_stiffness = FT(NaN), FT(NaN)
    for k in 1:(length(R_scan) - 1)
        F_scan[k] * F_scan[k + 1] < 0 || continue               # a sign change (NaN never is)
        root = FT(R_scan[k] - F_scan[k] * (R_scan[k + 1] - R_scan[k]) / (F_scan[k + 1] - F_scan[k]))
        if isnan(R_eq) || abs(root - R_cen_by_Jϕ) < abs(R_eq - R_cen_by_Jϕ)
            R_eq = root
        end
    end
    if !isnan(R_eq)
        R_eq_stiffness = (force_R_formula(R_eq + h) - force_R_formula(R_eq - h)) / (2h)
    end

    R_axis, Z_axis = _magnetic_axis(RP, closed)

    return (;
        I_closed, current_coherence, R_cen_by_Jϕ, Z_cen_by_Jϕ, a_eff_by_Jϕ, Bpol_edge,
        li, beta_p, pressure_column_share,
        Bv_req, BZ_ext_at_cen_by_Jϕ, BR_ext_at_cen_by_Jϕ, decay_index, decay_index_radial_limit,
        force_R_ext, force_Z_ext, force_R_coils, force_Z_coils, force_R_self, force_Z_self,
        force_R_pressure, force_R_toroidal, Bv_to_add,
        R_eq, R_eq_stiffness, R_axis, Z_axis,
        kappa_by_Jϕ, half_width_minor_by_Jϕ, half_width_major_by_Jϕ, force_R_pressure_all, Br_to_add,
        force_R_on_closed, force_Z_on_closed, Bv_to_add_closed,
        column_mass, nu_in_column, wall_gap_R, t_wall_inertial, t_wall_drag,
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

# On the grid row nearest Z, the distance from R to the last in-wall node in the direction of the
# radial force F (toward the outboard wall for F > 0, the inboard for F < 0; the nearer one at F = 0)
function _wall_gap_R(G::GridGeometry{FT}, R::FT, Z::FT, F::FT) where {FT <: AbstractFloat}
    j = clamp(round(Int, (Z - G.Z1D[1]) / G.dZ) + 1, 1, G.NZ)
    row = [G.R1D[i] for i in 1:G.NR if G.nodes.state[i, j] > 0.5]
    isempty(row) && return FT(NaN)
    inboard, outboard = R - minimum(row), maximum(row) - R
    F > 0 && return outboard
    F < 0 && return inboard
    return min(inboard, outboard)
end

# whether the run solves a self field: the boundary Green function and the Grad–Shafranov operator
# are built only with Ampère on
_solves_self_field(RP::RAPID) = !isempty(RP.G.Green_inWall2bdy)

# The plasma's own poloidal field: the Grad–Shafranov solve with its Jϕ alone as the source and its
# Green function on the boundary (the cached factorization, one back-substitution). NaN fields
# when the run solves no self field.
function _plasma_poloidal_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, OP, J = RP.G, RP.operators, RP.plasma.Jϕ
    BR, BZ = zeros(FT, G.NR, G.NZ), zeros(FT, G.NR, G.NZ)
    if !_solves_self_field(RP)
        fill!(BR, FT(NaN))
        fill!(BZ, FT(NaN))
        return BR, BZ
    end
    rhs = @. -RP.config.constants.μ0 * G.R2D * J
    rhs[G.BDY_idx] .= (G.Green_inWall2bdy * J[G.nodes.in_wall_nids]) .* (G.dR * G.dZ)
    OP.ΔGS_solver.F === nothing && factorize!(OP.ΔGS_solver, OP.ΔGS)
    ψ = similar(rhs)
    solve!(vec(ψ), OP.ΔGS_solver, vec(rhs))
    calculate_B_from_ψ!(G, ψ, BR, BZ)
    return BR, BZ
end

# the poloidal field of the coils the grid solves, at their present currents: zero without coils,
# NaN when the run solves no self field
function _coil_poloidal_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, csys = RP.G, RP.coil_system
    BR, BZ = zeros(FT, G.NR, G.NZ), zeros(FT, G.NR, G.NZ)
    csys.n_total > 0 || return BR, BZ
    if !_solves_self_field(RP)
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
