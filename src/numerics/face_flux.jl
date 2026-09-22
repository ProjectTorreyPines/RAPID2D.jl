# ∇·(u f) in flux form on cell faces, rows on in-wall nodes only.
#
#     F_{i+½} = u⁺_{i+½} f_i + u⁻_{i+½} f_{i+1},   u_{i+½} = ½(u_i + u_{i+1}),   u^± = ½(u ± |u|)
#     (∇·(u f))_i = [R_{i+½} F_{i+½} − R_{i−½} F_{i−½}] / (R_i ΔR)  +  [F_{j+½} − F_{j−½}] / ΔZ
#
# Every interior face is shared by exactly two rows with opposite signs, so Σ_i V_i (∇·(u f))_i
# telescopes to the wall faces alone — the property the nodal upwind `∇𝐮` lacks (no rows on
# the grid frame; a central-difference branch at |u| < eps that receives half of an upwind
# neighbour's outflow). A wall face — the neighbour across it is not in-wall — keeps only the
# owner's outflow term u⁺ f_i: nothing is read from outside, and the outflow sits on the
# diagonal, the convective counterpart of the Robin debit. `upwind = false` uses
# ½(f_i + f_{i+1}) on interior faces; wall faces stay upwind, since there is no f outside.
# internal/docs/src/notes/design/wall-flux-channels.md §2.6, §3.

"""
    build_face_flux_divergence(G, uR, uZ; upwind = true) -> SparseMatrixCSC

Matrix `A` with `(A f)_i = ∇·(u f)` at every in-wall node `i` and empty rows elsewhere,
built from face velocities. Interior faces average the two cell velocities; a wall face
uses the owner cell's velocity and drops the inflow term.
"""
function build_face_flux_divergence(
        G::GridGeometry{FT}, uR::AbstractMatrix{FT}, uZ::AbstractMatrix{FT};
        upwind::Bool = true,
    ) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    Ng = NR * NZ
    R = G.R2D
    inv_dR, inv_dZ = one(FT) / G.dR, one(FT) / G.dZ
    half = FT(0.5)
    I = Int[]
    Jc = Int[]
    V = FT[]
    nid(i, j) = (j - 1) * NR + i
    @inline function push_entry!(r, c, v)
        push!(I, r)
        push!(Jc, c)
        push!(V, v)
        return nothing
    end
    # One face of cell (i, j): `coef` carries the sign of the outward normal and the
    # face-area / cell-volume ratio; `un` is the velocity component leaving the cell.
    function add_face!(r, c_nb, neighbour_in, u_face, coef)
        un = coef * u_face   # sign(coef) = outward normal direction
        if neighbour_in && !upwind
            push_entry!(r, r, coef * u_face * half)
            push_entry!(r, c_nb, coef * u_face * half)
        elseif un > 0
            push_entry!(r, r, coef * u_face)          # outflow: the owner's value leaves
        elseif neighbour_in
            push_entry!(r, c_nb, coef * u_face)       # inflow from an interior neighbour
        end                                            # inflow through a wall face: nothing
        return nothing
    end
    for j in 1:NZ, i in 1:NR
        is_in_wall(G, i, j) || continue
        r = nid(i, j)
        for (di, sgn) in ((1, one(FT)), (-1, -one(FT)))
            ii = i + di
            nb_in = is_in_wall(G, ii, j)
            u_face = nb_in ? half * (uR[i, j] + uR[ii, j]) : uR[i, j]
            # face radius R ± ΔR/2 on every R-face, wall faces included — the same
            # A_f/V_i = R_face/(R_i ΔR) that `wall_faces` books, so the diagonal outflow
            # term and the ledger are one arithmetic
            R_face = R[i, j] + di * G.dR / 2
            add_face!(r, nb_in ? nid(ii, j) : 0, nb_in, u_face, sgn * R_face / R[i, j] * inv_dR)
        end
        for (dj, sgn) in ((1, one(FT)), (-1, -one(FT)))
            jj = j + dj
            nb_in = is_in_wall(G, i, jj)
            u_face = nb_in ? half * (uZ[i, j] + uZ[i, jj]) : uZ[i, j]
            add_face!(r, nb_in ? nid(i, jj) : 0, nb_in, u_face, sgn * inv_dZ)
        end
    end
    return sparse(I, Jc, V, Ng, Ng)
end

"""
    face_outflow_speeds(G, faces, uR, uZ) -> Vector

Convective absorption speed per wall face: the owner cell's velocity component through the
face, outflow only — `max(u·n̂, 0)`. Added to the diffusive `v_absorb` it makes one ledger
coefficient per face; the operator above charges exactly this on the diagonal.
"""
function face_outflow_speeds(
        G::GridGeometry{FT}, faces::AbstractVector{WallFace{FT}},
        uR::AbstractMatrix{FT}, uZ::AbstractMatrix{FT},
    ) where {FT <: AbstractFloat}
    v = Vector{FT}(undef, length(faces))
    for (k, f) in enumerate(faces)
        nR, nZ = f.outward
        un = nR * uR[f.rid, f.zid] + nZ * uZ[f.rid, f.zid]
        v[k] = max(zero(FT), un)
    end
    return v
end
