"""
    DiscretizedOperator{FT<:AbstractFloat}

A structure representing a discretized operator for numerical simulations.
This structure contains a sparse matrix representation of the operator,
along with its dimensions and a mapping from the original indices to the
corresponding indices in the sparse matrix.
# Fields:
- `matrix::SparseMatrixCSC{FT}`: Sparse matrix representation of the operator
- `dims::Tuple{Int,Int}`: Dimensions of the operator (number of rows and columns)
- `k2csc::Vector{Int}`: Mapping from original indices to sparse matrix indices
"""
function DiscretizedOperator{FT}(dims::Tuple{Int,Int}, I::Vector{Int}, J::Vector{Int}, V::Vector{FT}) where {FT<:AbstractFloat}
    dop = DiscretizedOperator{FT}(dims)

    oriV = copy(V)
    tmpV = FT.(1:length(V)) # to calculate k2csc

    dop.matrix = sparse(I, J, tmpV, prod(dims), prod(dims))

    # calculate k2csc
    dop.k2csc = zeros(Int, length(V))
    cscV = copy(dop.matrix.nzval)
    for csc_idx in eachindex(cscV)
        ori_k = round(Int, cscV[csc_idx])
        dop.k2csc[ori_k] = csc_idx
        # reset the nzvalue to the original value
        dop.matrix.nzval[csc_idx] = oriV[ori_k]
    end

    return dop
end

import Base: *

function Base.:(==)(dop1::DiscretizedOperator{FT1},
                    dop2::DiscretizedOperator{FT2}) where {FT1<:AbstractFloat, FT2<:AbstractFloat}
    return (dop1.dims == dop2.dims
            && dop1.matrix == dop2.matrix
            && dop1.k2csc == dop2.k2csc)
end

function *(dop::DiscretizedOperator{FT}, B::AbstractMatrix{FT}) where {FT<:AbstractFloat}
    return reshape(dop.matrix * @view(B[:]), dop.dims)
end

function *(dop::DiscretizedOperator{FT}, B::AbstractVector{FT}) where {FT<:AbstractFloat}
    return dop.matrix * B
end
