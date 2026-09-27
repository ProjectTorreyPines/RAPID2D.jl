# The one sparsity pattern of the in-wall operators, and in-place arithmetic on it.
#
# Every in-wall operator (face-flux divergence, 9-point diffusion, advection, the θ-matrices
# built from them) lives on this pattern as a `DiscretizedOperator`: the 9-point stencil on
# in-wall neighbours, plus the diagonal on every node so that `I − θΔt·A` fits too. Both upwind
# sides of every face are in it, so the structure is the same for any velocity field; values are
# written in place and the side a flow does not use is a stored zero. `k2csc` maps (row, stencil
# slot) to the position in `nonzeros`: `k2csc[9(row − 1) + slot]`, 0 where the slot is absent —
# the retired nodal `k2csc` with a fixed stride of nine, since in-wall rows have as many
# neighbours as the wall leaves them.

const STENCIL_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
const SLOT_C, SLOT_E, SLOT_W, SLOT_N, SLOT_S, SLOT_NE, SLOT_NW, SLOT_SE, SLOT_SW = 1, 2, 3, 4, 5, 6, 7, 8, 9
# SLOT_LOOKUP[di + 2, dj + 2] is the slot of offset (di, dj)
const SLOT_LOOKUP = [SLOT_SW SLOT_W SLOT_NW; SLOT_S SLOT_C SLOT_N; SLOT_SE SLOT_E SLOT_NE]
@inline stencil_slot(di::Int, dj::Int) = @inbounds SLOT_LOOKUP[di + 2, dj + 2]

# the nzval position of stencil slot `s` in row `r` (0 if absent)
@inline slot_position(k2csc::Vector{Int}, r::Int, s::Int) = @inbounds k2csc[9 * (r - 1) + s]

# a column's rows in increasing order: row = c + dj·NR + di with NR ≥ 3
const COLUMN_ORDER = ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

"""
    build_wall_pattern(G) -> DiscretizedOperator

A zero operator on the in-wall pattern of `G` (see the file header), built directly in CSC:
the pattern is structurally symmetric, so column `c` holds `c`'s own stencil rows, already in
increasing order. Values are allocated as zeros; `similar` gives further operators on it.
"""
function build_wall_pattern(G::GridGeometry{FT}) where {FT <: AbstractFloat}
    NR, NZ = G.NR, G.NZ
    NR >= 3 || throw(ArgumentError("the wall pattern needs NR ≥ 3 (got $NR)"))
    Ng = NR * NZ
    nid = G.nodes.nid
    colptr = Vector{Int}(undef, Ng + 1)
    rowval = Int[]
    sizehint!(rowval, 9 * Ng)
    k2csc = zeros(Int, 9 * Ng)
    colptr[1] = 1
    for j in 1:NZ, i in 1:NR
        c = nid[i, j]
        c_in = is_in_wall(G, i, j)
        for (di, dj) in COLUMN_ORDER
            if di == 0 && dj == 0
                push!(rowval, c)
                k2csc[9 * (c - 1) + SLOT_C] = length(rowval)
            elseif c_in && is_in_wall(G, i + di, j + dj)
                r = nid[i + di, j + dj]
                push!(rowval, r)
                # entry (r, c): from row r, column c sits at offset (−di, −dj)
                k2csc[9 * (r - 1) + stencil_slot(-di, -dj)] = length(rowval)
            end
        end
        colptr[c + 1] = length(rowval) + 1
    end
    M = SparseMatrixCSC(Ng, Ng, colptr, rowval, zeros(FT, length(rowval)))
    return DiscretizedOperator{FT}(dims_rz = (NR, NZ), matrix = M, k2csc = k2csc)
end

"Throw unless `A` lives on a wall pattern (`k2csc` with nine slots per row)."
function check_wall_pattern(A::DiscretizedOperator)
    Ng = prod(A.dims_rz)
    (size(A.matrix) == (Ng, Ng) && length(A.k2csc) == 9 * Ng) || throw(
        ArgumentError(
            "the operator is not allocated on the wall pattern " *
                "(use build_wall_pattern(G) or similar of an operator on it)"
        )
    )
    return A
end

"`A ← I` on the wall pattern (every row has its diagonal)."
function set_identity!(A::DiscretizedOperator{FT}) where {FT <: AbstractFloat}
    check_wall_pattern(A)
    nz, k2c = nonzeros(A.matrix), A.k2csc
    fill!(nz, zero(FT))
    @inbounds for r in 1:size(A.matrix, 1)
        nz[slot_position(k2c, r, SLOT_C)] = one(FT)
    end
    return A
end

"`A ← A + c·B` for two operators on one pattern; nothing is dropped, so the structure stays."
function add_scaled!(A::DiscretizedOperator{FT}, c, B::DiscretizedOperator{FT}) where {FT <: AbstractFloat}
    MA, MB = A.matrix, B.matrix
    (size(MA) == size(MB) && MA.colptr == MB.colptr && MA.rowval == MB.rowval) ||
        throw(ArgumentError("add_scaled!: the two operators do not share a pattern"))
    cf = FT(c)
    nzA, nzB = nonzeros(MA), nonzeros(MB)
    @inbounds @simd for k in eachindex(nzA)
        nzA[k] += cf * nzB[k]
    end
    return A
end

"`A ← A + scale·diag(v)` on the wall pattern, `v` indexed by node."
function add_diagonal!(A::DiscretizedOperator{FT}, v::AbstractVector; scale = one(FT)) where {FT <: AbstractFloat}
    check_wall_pattern(A)
    Ng = size(A.matrix, 1)
    length(v) == Ng || throw(DimensionMismatch("add_diagonal!: got $(length(v)) values for $Ng nodes"))
    nz, k2c = nonzeros(A.matrix), A.k2csc
    s = FT(scale)
    @inbounds for r in 1:Ng
        nz[slot_position(k2c, r, SLOT_C)] += s * v[r]
    end
    return A
end
