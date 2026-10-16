# The current column's zeroth-order force balance: what a reduced position model or an ideal
# vertical-field controller needs from the 2D state. Measured at the 0D snapshots only.

# the names `column_force_balance` returns, in the order the snapshot lists them
const COLUMN_FORCE_BALANCE_FIELDS = (
    :I_closed, :R_centroid, :Z_centroid, :a_eff, :li, :beta_p,
    :Bv_req, :BZ_ext_centroid, :BR_ext_centroid, :decay_index, :decay_index_radial_limit,
    :force_R_ext, :force_Z_ext, :force_R_self, :force_Z_self, :force_R_pressure, :force_R_toroidal,
    :Bv_to_add, :R_eq, :R_eq_stiffness,
)

"""
    column_force_balance(RP) -> NamedTuple

The toroidal current column's zeroth-order force balance, a **total-current circular screening
estimate**: the whole current is taken as one circular column at its centroid.

- **Column.** `I_closed`: the current on closed field lines. `R_centroid`, `Z_centroid`: the
  |Jϕ|-weighted centroid. `a_eff = √(2⟨r²⟩)` about it, which gives back the radius of a uniform
  disc. `li` and `beta_p` inside that disc, from `p = e (nₑ Tₑ + nᵢ Tᵢ)` and the self field less
  its mean over the disc (the column's own vertical field, which an equilibrium cancels).
- **Shafranov.** `Bv_req = −μ0 I Λ / (4π R)`, `Λ = ln(8R/a) + β_p + l_i/2 − 3/2`: the vertical
  field a circular column at the centroid needs. `BZ_ext_centroid` and `BR_ext_centroid`: what
  the external field gives there. `decay_index = −(R/B_Z) ∂B_Z/∂R` of the external field there.
  Vertical stability needs it above 0, radial stability at fixed current below
  `decay_index_radial_limit = 1 − 1/Λ` (at fixed flux, below about 3/2).
- **Forces [N]** from the model's own fields, with `dV = 2πR dR dZ`:
  - `force_R_ext = ∫ Jϕ B_Z,ext dV` and `force_Z_ext = −∫ Jϕ B_R,ext dV`;
  - `force_R_self`, `force_Z_self`: the same with the self field, the plasma's and that of the
    coils the grid solves. Its radial part is the hoop force;
  - `force_R_pressure = 2π ∫ p dR dZ`, the tire-tube force;
  - `force_R_toroidal = (μ0 I²/4)(β_p − 1)`, the toroidal-field force of a column in minor-radius
    equilibrium. The model does not carry it (it has no δB_ϕ); the Shafranov Λ does.
- **Missing vertical field.** `Bv_to_add = −F_R / ∫ Jϕ dV`, with `F_R` the sum of the four radial
  forces: the uniform B_Z that, added, zeroes the net radial force. It is what an ideal
  vertical-field controller would apply. `Bv_req − BZ_ext_centroid` estimates the same from the
  formula.
- **Equilibrium radius.** `R_eq`: on the centroid's midplane, the radius nearest the centroid
  where `F_R(R) = (μ0 I²/2) Λ(R) + 2πR I B_Z,ext(R)` vanishes, with `a`, `β_p`, `l_i` held.
  `R_eq_stiffness = dF_R/dR` there: negative is restoring.

All are zero below `flags.Ampere_Itor_threshold` of current, and `R_eq`, `R_eq_stiffness` also
where `F_R(R)` has no root.
"""
function column_force_balance(RP::RAPID{FT}) where {FT <: AbstractFloat}
    G, F, pla = RP.G, RP.fields, RP.plasma
    μ0, ee = RP.config.constants.μ0, RP.config.constants.ee
    J = pla.Jϕ
    dA = G.dR * G.dZ
    I = sum(J) * dA
    if iszero(I) || abs(I) < RP.flags.Ampere_Itor_threshold
        return NamedTuple{COLUMN_FORCE_BALANCE_FIELDS}(ntuple(_ -> zero(FT), length(COLUMN_FORCE_BALANCE_FIELDS)))
    end
    V = G.inVol2D

    # the column: centroid, effective radius, and l_i, β_p inside it
    closed = [k for k in RP.flf.closed_surface_nids if G.nodes.state[k] > 0.5]
    I_closed = sum(J[k] for k in closed; init = zero(FT)) * dA
    weight = abs.(J)
    R_centroid = sum(weight .* G.R2D) / sum(weight)
    Z_centroid = sum(weight .* G.Z2D) / sum(weight)
    r² = @. (G.R2D - R_centroid)^2 + (G.Z2D - Z_centroid)^2
    a_eff = max(sqrt(2 * sum(weight .* r²) / sum(weight)), sqrt(dA / π))   # at least one cell's disc
    inside = r² .<= a_eff^2
    # l_i from the self field less its mean over the disc: that uniform part is the column's own
    # vertical field (its hoop force), which the vertical field of an equilibrium cancels
    V_in = V[inside]
    BR_in, BZ_in = F.BR_self[inside], F.BZ_self[inside]
    BR_in .-= sum(BR_in .* V_in) / sum(V_in)
    BZ_in .-= sum(BZ_in .* V_in) / sum(V_in)
    li = 2 * sum(@. (BR_in^2 + BZ_in^2) * V_in) / (μ0^2 * I^2 * R_centroid)
    p = @. ee * (pla.ne * pla.Te_eV + pla.ni * pla.Ti_eV)
    beta_p = 4 * sum(p[inside] .* V_in) / (μ0 * I^2 * R_centroid)

    # Shafranov: the vertical field the column needs, and the external field it has
    Λ(R) = log(8R / a_eff) + beta_p + li / 2 - FT(1.5)
    Bv_req = -μ0 * I * Λ(R_centroid) / (4π * R_centroid)
    BZ_ext_centroid = _bilinear(G, F.BZ_ext, R_centroid, Z_centroid)
    BR_ext_centroid = _bilinear(G, F.BR_ext, R_centroid, Z_centroid)
    h = G.dR / 2
    dBZ_dR = (_bilinear(G, F.BZ_ext, R_centroid + h, Z_centroid) - _bilinear(G, F.BZ_ext, R_centroid - h, Z_centroid)) / (2h)
    decay_index = iszero(BZ_ext_centroid) ? zero(FT) : -R_centroid / BZ_ext_centroid * dBZ_dR
    decay_index_radial_limit = one(FT) - one(FT) / Λ(R_centroid)

    # the forces of the model's own fields, and the toroidal-field force it does not carry
    force_R_ext = sum(J .* F.BZ_ext .* V)
    force_Z_ext = -sum(J .* F.BR_ext .* V)
    force_R_self = sum(J .* F.BZ_self .* V)
    force_Z_self = -sum(J .* F.BR_self .* V)
    force_R_pressure = sum(p .* V ./ G.R2D)                  # 2π ∫ p dR dZ
    force_R_toroidal = μ0 * I^2 / 4 * (beta_p - 1)
    force_R_net = force_R_ext + force_R_self + force_R_pressure + force_R_toroidal
    Bv_to_add = -force_R_net / sum(J .* V)

    # the equilibrium radius of the formula on the centroid's midplane, nearest the centroid
    force_R_formula(R) = μ0 * I^2 / 2 * Λ(R) + 2π * R * I * _bilinear(G, F.BZ_ext, R, Z_centroid)
    R_scan = range(G.R1D[2], G.R1D[end - 1]; length = 4 * G.NR)
    F_scan = force_R_formula.(R_scan)
    R_eq, R_eq_stiffness = zero(FT), zero(FT)
    for k in 1:(length(R_scan) - 1)
        sign(F_scan[k]) == sign(F_scan[k + 1]) && continue
        root = R_scan[k] - F_scan[k] * (R_scan[k + 1] - R_scan[k]) / (F_scan[k + 1] - F_scan[k])
        if iszero(R_eq) || abs(root - R_centroid) < abs(R_eq - R_centroid)
            R_eq = FT(root)
        end
    end
    if !iszero(R_eq)
        R_eq_stiffness = (force_R_formula(R_eq + h) - force_R_formula(R_eq - h)) / (2h)
    end

    return (;
        I_closed, R_centroid, Z_centroid, a_eff, li, beta_p,
        Bv_req, BZ_ext_centroid, BR_ext_centroid, decay_index, decay_index_radial_limit,
        force_R_ext, force_Z_ext, force_R_self, force_Z_self, force_R_pressure, force_R_toroidal,
        Bv_to_add, R_eq, R_eq_stiffness,
    )
end

# the bilinear value of a grid field at (R, Z), clamped to the grid
function _bilinear(G::GridGeometry{FT}, field::AbstractMatrix{FT}, R::Real, Z::Real) where {FT <: AbstractFloat}
    x = (R - G.R1D[1]) / G.dR + 1
    y = (Z - G.Z1D[1]) / G.dZ + 1
    i = clamp(floor(Int, x), 1, G.NR - 1)
    j = clamp(floor(Int, y), 1, G.NZ - 1)
    s, t = x - i, y - j
    return FT((1 - s) * (1 - t) * field[i, j] + s * (1 - t) * field[i + 1, j] + (1 - s) * t * field[i, j + 1] + s * t * field[i + 1, j + 1])
end
