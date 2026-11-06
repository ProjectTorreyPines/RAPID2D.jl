# The open lines' circuit (`notes/plans/PLAN_open-line-circuit.md`): along each open line that ends on
# the wall both ways the parallel current per flux, J∥/B, is one value, the residual field is J∥ η, and
# the line's EMF against its resistance and its end condition set that value. The shielding of the
# driven E∥ is a result, γ = 1 − E_res/E_drive, negative where the EMF piles up on a resistive stretch.

# below this density a node holds no electrons: the resistivity's floor
const OPEN_LINE_N_FLOOR = 1.0

"""
    driven_parallel_field(RP) -> Matrix

The driven E∥: the prescribed field's, the vertical-field control's (in a run that has one) and the
induced one (with `E_para_induction`); what the sheets, or the line circuit, shield.
"""
function driven_parallel_field(RP::RAPID{FT}) where {FT <: AbstractFloat}
    F = RP.fields
    EM = RP.flags.E_para_induction
    return if _vertical_field_control_on(RP)
        @. F.Eϕ_para_presc + F.Eϕ_para_ctrl + EM * F.Eϕ_para_coils_pla
    else
        @. F.Eϕ_para_presc + EM * F.Eϕ_para_coils_pla
    end
end

"""
    parallel_resistivity(RP) -> Matrix

η = mₑ ν_eff/(nₑ e²), ν_eff the momentum solve's total rate (e–i, e–n momentum transfer and the
newborn electrons' dilution), as the 2D snapshots record it; at the floor density where there are no
electrons.
"""
function parallel_resistivity(RP::RAPID{FT}) where {FT <: AbstractFloat}
    pla = RP.plasma
    @unpack me, ee = RP.config.constants
    return @. me * (pla.ν_ei_eff + pla.ν_en_mom_tot + pla.ν_en_iz_tot) / (max(pla.ne, FT(OPEN_LINE_N_FLOOR)) * ee^2)
end

# The wall faces owned by each node, by node id.
function faces_by_node(faces::AbstractVector{<:WallFace})
    by = Dict{Int, Vector{Int}}()
    for (k, f) in enumerate(faces)
        push!(get!(by, f.nid, Int[]), k)
    end
    return by
end

# The wall face a line leaves through at its end (R, Z) along the direction `sgn` b̂_pol: among the
# faces of the nearest node and its neighbours, the one facing that direction most; 0 without one.
function end_face(G::GridGeometry{FT}, faces, by_node, R, Z, bR, bZ, sgn) where {FT}
    (isfinite(R) && isfinite(Z)) || return 0
    i = clamp(round(Int, (R - G.R1D[1]) / G.dR) + 1, 1, G.NR)
    j = clamp(round(Int, (Z - G.Z1D[1]) / G.dZ) + 1, 1, G.NZ)
    best, best_dot = 0, -FT(Inf)
    for dj in -1:1, di in -1:1
        ii, jj = i + di, j + dj
        (1 <= ii <= G.NR && 1 <= jj <= G.NZ) || continue
        nid = (jj - 1) * G.NR + ii
        for k in get(by_node, nid, Int[])
            nR, nZ = faces[k].outward
            d = sgn * (nR * bR[ii, jj] + nZ * bZ[ii, jj]) - (abs(di) + abs(dj)) * FT(1.0e-3)   # the node itself first
            if d > best_dot
                best, best_dot = k, d
            end
        end
    end
    return best
end

# The current per flux the end policy lets through: the ion-collecting end's wall supplies at most
# its ions' charge flux per flux-tube cross-section, Σ_s Z_s Γ_s / μ, μ = |b̂·n̂_f| the line's incidence
Jpara_over_B_limit(::ConductingEnds, args...) = Inf
Jpara_over_B_limit(::CappedEnds, args...) = Inf
function Jpara_over_B_limit(::IonSupplyEnds, st::OpenLineCircuitState{FT}, faces, k_face, μ, B_end) where {FT}
    k_face == 0 && return FT(Inf)
    μ > 0 || return FT(Inf)
    return st.ion_wall_charge_flux[k_face] / (μ * B_end)
end

# The line's current per flux from its EMF, resistance and limit
function line_current(::Union{ConductingEnds, IonSupplyEnds}, V::FT, Rl::FT, limit::FT) where {FT}
    Rl > 0 || return zero(FT)
    return sign(V) * min(abs(V) / Rl, limit)
end
function line_current(ends::CappedEnds, V::FT, Rl::FT, limit::FT) where {FT}
    Rl > 0 || return zero(FT)
    return (V - sign(V) * min(abs(V), FT(ends.Φ))) / Rl
end

"""
    open_line_circuit!(RP) -> RP

The line circuit on every node of an open line that ends on the wall both ways (`flf.line_weights`,
`flf.end_*`): the line EMF `∫ E_drive dl`, the resistance `∫ B η dl` over the plasma part of the line,
the current per flux `Jpara_over_B` the end policy allows, the residual field `E_para_res = Jpara_over_B B η`
and the shielding `gamma_local`, into `RP.open_line`. Nothing without `LineCircuit`.
"""
function open_line_circuit!(RP::RAPID{FT}) where {FT <: AbstractFloat}
    policy = RP.flags.open_line_shielding
    st, F, G, flf = RP.open_line, RP.fields, RP.G, RP.flf
    fill!(st.applies, false)
    for A in (st.line_emf, st.line_resistance, st.Jpara_over_B, st.Jpara_over_B_limit, st.E_para_res, st.gamma_local)
        fill!(A, FT(NaN))
    end
    policy isa LineCircuit || return RP
    W = flf.line_weights
    nnz(W) == 0 && return RP
    # The line's EMF and resistance over its plasma part alone: nodes below `n_cut` of the densest
    # (the gap between the plasma and the wall, where the sheets do not shield) carry neither, a
    # declared approximation (PLAN_open-line-circuit.md §7).
    ne = RP.plasma.ne
    plasma = ne .> policy.n_cut * maximum(ne[G.nodes.in_wall_nids])
    E_drive = driven_parallel_field(RP)
    η = parallel_resistivity(RP)
    Bη = F.Btot .* η
    V = reshape(W * vec(E_drive .* plasma), size(E_drive))
    Rl = reshape(W * vec(Bη .* plasma), size(Bη))
    faces = RP.transport.wall_faces
    by_node = faces_by_node(faces)
    bR, bZ = F.bR, F.bZ
    rows = unique(rowvals(W))     # the nodes with a row: a line that ends on the wall both ways
    for k in rows
        i, j = mod1(k, G.NR), cld(k, G.NR)
        plasma[i, j] || continue
        # the ion-collecting end: ions leave along J, along +b̂ for a positive current
        Vk, Rk = V[i, j], Rl[i, j]
        forward = Vk >= 0
        kf = forward ? end_face(G, faces, by_node, flf.end_R_forward[i, j], flf.end_Z_forward[i, j], bR, bZ, 1) :
            end_face(G, faces, by_node, flf.end_R_backward[i, j], flf.end_Z_backward[i, j], bR, bZ, -1)
        μ, B_end = zero(FT), F.Btot[i, j]
        if kf != 0
            f = faces[kf]
            μ = abs(f.outward[1] * bR[f.rid, f.zid] + f.outward[2] * bZ[f.rid, f.zid])
            B_end = F.Btot[f.rid, f.zid]
        end
        limit = FT(Jpara_over_B_limit(policy.ends, st, faces, kf, μ, B_end))
        K = line_current(policy.ends, Vk, Rk, limit)
        st.applies[i, j] = true
        st.line_emf[i, j], st.line_resistance[i, j] = Vk, Rk
        st.Jpara_over_B[i, j], st.Jpara_over_B_limit[i, j] = K, limit
        st.E_para_res[i, j] = K * Bη[i, j]
        st.gamma_local[i, j] = E_drive[i, j] == 0 ? FT(NaN) : 1 - st.E_para_res[i, j] / E_drive[i, j]
    end
    return RP
end

"""
    circuit_wall_speeds!(RP, faces, v_absorb, v_out, albedo) -> (v_absorb, albedo)

The electrons' wall speeds a face takes under the line circuit: on a face whose owner node is on a
line the circuit applies to, the sheath passes the ions and turns the electrons back, so the
electrons leave at the rate the circuit leaves them, `Γ_e = Γ_i − j_out/e` per wall area, with
`j_out = Jpara_over_B B·n̂_f` the current into the wall (positive at the ion-collecting end, where
fewer electrons leave; negative at the other, where more do) and `Γ_i` the ions' charge flux of
their last step. The target speed `Γ_e/nₑ` (at least zero) is met by the face's albedo, which returns
the convective outflow `v_out` the sheath turns back, and by the Robin speed `v_absorb` for the rest;
the ledger reads the same speeds. Other faces keep the channels' speeds and the configured albedo.
With the convective channel alone, a target above the outflow is not reached. The electrons'
density is the step's entry value: a θ-weighted solve lands near the target, not on it
(PLAN_open-line-circuit.md §7).
"""
function circuit_wall_speeds!(
        RP::RAPID{FT}, faces, v_absorb::AbstractVector{FT}, v_out::AbstractVector{FT}, albedo::AbstractVector{FT},
    ) where {FT <: AbstractFloat}
    RP.flags.open_line_shielding isa LineCircuit || return v_absorb, albedo
    st, F, pla = RP.open_line, RP.fields, RP.plasma
    ee = RP.config.constants.ee
    diffu, convec = RP.flags.diffu, RP.flags.convec
    for (k, f) in enumerate(faces)
        st.applies[f.rid, f.zid] || continue
        n = pla.ne[f.rid, f.zid]
        n > OPEN_LINE_N_FLOOR || continue
        nR, nZ = f.outward
        j_out = st.Jpara_over_B[f.rid, f.zid] * (F.BR[f.rid, f.zid] * nR + F.BZ[f.rid, f.zid] * nZ)
        v_target = max(zero(FT), (st.ion_wall_charge_flux[k] - j_out) / (ee * n))
        if convec && v_out[k] > 0
            # the outflow the sheath lets through, the rest returned; what is still missing, the Robin speed
            through = min(v_target, v_out[k])
            albedo[k] = one(FT) - through / v_out[k]
            v_absorb[k] = diffu ? v_target - through : zero(FT)
        else
            albedo[k] = convec ? one(FT) : albedo[k]
            v_absorb[k] = diffu ? v_target : v_absorb[k]
        end
    end
    return v_absorb, albedo
end
