# The coupled solve's u∥–ψ block matrix, on a pattern fixed once and written in place.

"""
    CoupledBlock(A_u, A_GS)

The 2N × 2N matrix of the coupled solve,

    [ A_u          diag(c_uψ) ]
    [ diag(c_ψu)   A_GS       ],

on a pattern fixed at construction: `A_u`'s (the wall pattern, which stores both upwind sides of
every face), `A_GS`'s, and the two coupling diagonals on every node. [`update_coupled_block!`](@ref)
writes values only, so a cached LU keeps its symbolic analysis whichever way the flow turns.
"""
struct CoupledBlock{FT <: AbstractFloat}
    matrix::SparseMatrixCSC{FT, Int}
    pos_u::Vector{Int}    # position in nonzeros(matrix) of each stored entry of A_u
    pos_GS::Vector{Int}   # of each stored entry of A_GS, in the lower-right block
    pos_uψ::Vector{Int}   # of entry (i, N + i)
    pos_ψu::Vector{Int}   # of entry (N + i, i)
end

function CoupledBlock(A_u::DiscretizedOperator{FT}, A_GS::DiscretizedOperator{FT}) where {FT <: AbstractFloat}
    Mu, Mg = A_u.matrix, A_GS.matrix
    N = size(Mu, 1)
    size(Mg) == (N, N) || throw(DimensionMismatch("A_u is $(size(Mu)) but A_GS is $(size(Mg))"))
    Iu, Ju, _ = findnz(Mu)   # stored entries, zeros included, in nonzeros order
    Ig, Jg, _ = findnz(Mg)
    rows = vcat(Iu, Ig .+ N, 1:N, (N + 1):(2N))
    cols = vcat(Ju, Jg .+ N, (N + 1):(2N), 1:N)
    M = sparse(rows, cols, ones(FT, length(rows)), 2N, 2N)
    fill!(nonzeros(M), zero(FT))
    pos(r, c) = first(nzrange(M, c)) + searchsortedfirst(view(rowvals(M), nzrange(M, c)), r) - 1
    return CoupledBlock{FT}(
        M, pos.(Iu, Ju), pos.(Ig .+ N, Jg .+ N), [pos(i, N + i) for i in 1:N], [pos(N + i, i) for i in 1:N],
    )
end

"""
    update_coupled_block!(B::CoupledBlock, A_u, A_GS, c_uψ, c_ψu) -> B.matrix

Write the values of `A_u`, `A_GS` and the coupling diagonals into `B`; the structure stays.
`A_u` and `A_GS` must be on the patterns `B` was built from.
"""
function update_coupled_block!(
        B::CoupledBlock{FT}, A_u::DiscretizedOperator{FT}, A_GS::DiscretizedOperator{FT},
        c_uψ::AbstractVector, c_ψu::AbstractVector,
    ) where {FT <: AbstractFloat}
    nzu, nzg = nonzeros(A_u.matrix), nonzeros(A_GS.matrix)
    (length(nzu) == length(B.pos_u) && length(nzg) == length(B.pos_GS)) ||
        throw(ArgumentError("update_coupled_block!: A_u or A_GS is not on the pattern the block was built from"))
    nz = nonzeros(B.matrix)
    @inbounds begin
        for k in eachindex(nzu)
            nz[B.pos_u[k]] = nzu[k]
        end
        for k in eachindex(nzg)
            nz[B.pos_GS[k]] = nzg[k]
        end
        for i in eachindex(B.pos_uψ)
            nz[B.pos_uψ[i]] = c_uψ[i]
            nz[B.pos_ψu[i]] = c_ψu[i]
        end
    end
    return B.matrix
end
