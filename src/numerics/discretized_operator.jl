export DiscretizedOperator

"""
    DiscretizedOperator{FT<:AbstractFloat}

Represents a discretized operator in a two-dimensional domain.

# Fields
- `dims::Tuple{Int,Int}`: The dimensions (NR, NZ) of the discretized domain
- `matrix::SparseMatrixCSC{FT,Int}`: Sparse matrix representation of the discretized operator
- `k2csc::Vector{Int}`: Mapping from k-indices to CSC indices for efficient sparse matrix updates
"""
@kwdef mutable struct DiscretizedOperator{FT<:AbstractFloat}
    dims_rz::Tuple{Int,Int} # (NR, NZ)

    # sparse matrix for the discretized operator
    matrix::SparseMatrixCSC{FT,Int} = spzeros(FT, prod(dims_rz), prod(dims_rz))

    # Mapping from k-index to CSC index (not always used)
    # (for more efficient update of non-zero elements of CSC matrix)
    k2csc::Vector{Int} = Int[]
end

function DiscretizedOperator{FT}(dimensions::Tuple{Int,Int}) where {FT<:AbstractFloat}
    return DiscretizedOperator{FT}(dims_rz=dimensions)
end

function DiscretizedOperator{FT}(NR::Int, NZ::Int) where {FT<:AbstractFloat}
    return DiscretizedOperator{FT}(dims_rz=(NR, NZ))
end

function DiscretizedOperator(dims_rz::Tuple{Int,Int}, I::Vector{Int}, J::Vector{Int}, V::Vector{FT}) where {FT<:AbstractFloat}
    dop = DiscretizedOperator{FT}(dims_rz)

    oriV = copy(V)
    tmpV = FT.(1:length(V)) # to calculate k2csc

    dop.matrix = sparse(I, J, tmpV, prod(dims_rz), prod(dims_rz))

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

import Base: size, axes, IndexStyle
size(op::DiscretizedOperator) = size(op.matrix)
axes(op::DiscretizedOperator) = axes(op.matrix)
IndexStyle(op::DiscretizedOperator) = IndexStyle(op.matrix)

import Base: +, -, *, /, ^, inv, ==
import Base: materialize, BroadcastStyle, broadcastable, broadcasted
import Base.Broadcast: Broadcasted, combine_styles
import Base: broadcast
# 2. Define a custom broadcast style for DiscretizedOperator
struct DOStyle <: BroadcastStyle end
BroadcastStyle(::Type{<:DiscretizedOperator}) = DOStyle()


function Base.:(==)(dop1::DiscretizedOperator{FT1},
                    dop2::DiscretizedOperator{FT2}) where {FT1<:AbstractFloat, FT2<:AbstractFloat}
    return (dop1.dims_rz == dop2.dims_rz
            && dop1.matrix == dop2.matrix
            && dop1.k2csc == dop2.k2csc)
end

function *(dop::DiscretizedOperator{FT}, mat::AbstractMatrix{FT}) where {FT<:AbstractFloat}
    return reshape(dop.matrix * @view(mat[:]), dop.dims_rz)
end

function *(dop::DiscretizedOperator{FT}, vec::AbstractVector{FT}) where {FT<:AbstractFloat}
    return dop.matrix * vec
end

# Unary negation & inverse —
-(A::DiscretizedOperator) = DiscretizedOperator(dims_rz = A.dims_rz, matrix = -A.matrix)
inv(A::DiscretizedOperator)  = DiscretizedOperator(dims_rz = A.dims_rz, matrix = inv(A.matrix))

# — Binary combination (same dims only) —
function +(A::DiscretizedOperator, B::DiscretizedOperator)
    A.dims_rz == B.dims_rz || throw(DimensionMismatch("dims_rz of A=$(A.dims_rz) and B=$(B.dims_rz) do not match"))
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix + B.matrix)
end
function +(A::DiscretizedOperator, B::AbstractMatrix)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix + B)
end
function +(A::AbstractMatrix, B::DiscretizedOperator)
    DiscretizedOperator(dims_rz = B.dims_rz, matrix = A + B.matrix)
end

function -(A::DiscretizedOperator, B::DiscretizedOperator)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix - B.matrix)
end
function -(A::DiscretizedOperator, B::AbstractMatrix)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix - B)
end
function -(A::AbstractMatrix, B::DiscretizedOperator)
    DiscretizedOperator(dims_rz = B.dims_rz, matrix = A - B.matrix)
end

function *(A::DiscretizedOperator, B::DiscretizedOperator)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix * B.matrix)
end
function *(A::DiscretizedOperator, B::AbstractMatrix)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix * B)
end
function *(A::AbstractMatrix, B::DiscretizedOperator)
    DiscretizedOperator(dims_rz = B.dims_rz, matrix = A * B.matrix)
end


# — Scalar * operator and operator * scalar —
*(α::Number, A::DiscretizedOperator) = DiscretizedOperator(dims_rz = A.dims_rz, matrix = α * A.matrix)
*(A::DiscretizedOperator, α::Number) = DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix * α)

# — Division by scalar and exponentiation —
function /(A::DiscretizedOperator, α::Number)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix / α)
end
function /(α::Number, A::DiscretizedOperator)
    DiscretizedOperator(dims_rz = A.dims_rz, matrix = α / A.matrix )
end
^(A::DiscretizedOperator, n::Integer) = DiscretizedOperator(dims_rz = A.dims_rz, matrix = A.matrix^n)

## === Copy operator ===
import Base: copyto!

function Base.copy(src::DiscretizedOperator{FT}) where {FT<:AbstractFloat}
    new_op = DiscretizedOperator{FT}(dims_rz = src.dims_rz) # Assigns the tuple
    new_op.matrix = copy(src.matrix) # Deep copy the matrix
    new_op.k2csc = copy(src.k2csc)   # Deep copy the vector
    return new_op
end

function copyto!(dest::DiscretizedOperator{FT}, src::DiscretizedOperator{FT}) where {FT<:AbstractFloat}
	@assert dest.dims_rz == src.dims_rz "Dimensions of dest=$(dest.dims_rz) and src=$(src.dims_rz) do not match"
    copyto!(dest.matrix, src.matrix)
	dest.k2csc = copy(src.k2csc)
	return dest
end

function copyto!(dest::DiscretizedOperator{FT}, src::AbstractMatrix{FT}) where {FT<:AbstractFloat}
    copyto!(dest.matrix, src)
	empty!(dest.k2csc) # reset k2csc
	return dest
end

# Special Broadcasted type handler for .= operations
function copyto!(dest::DiscretizedOperator{FT}, bc::Broadcasted{DOStyle}) where {FT}
	src_dops = filter(x -> x isa DiscretizedOperator, bc.args)
    @assert !isempty(src_dops) "No DiscretizedOperator found in broadcast args"
    src = first(src_dops)
    @assert src.dims_rz == dest.dims_rz "Dimensions of dest=$(dest.dims_rz) and src=$(src.dims_rz) do not match"

    vals = map(bc.args) do x
        x isa DiscretizedOperator ? x.matrix : x
    end
    bc_mat = broadcasted(bc.f, vals...)
    M = materialize(bc_mat)

    copyto!(dest.matrix, M)
	dest.k2csc = copy(src.k2csc)
    return dest
end

# General Broadcasted type handler for .= operations
function copyto!(dest::DiscretizedOperator{FT}, bc::Base.Broadcast.Broadcasted) where {FT}
    # Materialize the broadcasted operation to get the actual result
    result = Base.materialize(bc)

    # Handle the result based on its type
    if result isa DiscretizedOperator
        @assert dest.dims_rz == result.dims_rz "Dimensions must match: dest=$(dest.dims_rz), src=$(result.dims_rz)"
        copyto!(dest.matrix, result.matrix)
        dest.k2csc = copy(result.k2csc)
    elseif result isa AbstractMatrix
        copyto!(dest.matrix, result)
        empty!(dest.k2csc)  # Reset k2csc mapping
    else
        error("Cannot copy $(typeof(result)) to DiscretizedOperator")
    end

    return dest
end


## === Backlash operator for solving linear systems ===
import Base: \
function \(A::DiscretizedOperator{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    return A.matrix \ b
end
function \(A::DiscretizedOperator{T}, b::AbstractMatrix{T}) where {T<:AbstractFloat}
    A.dims_rz == size(b) || throw(DimensionMismatch("A.dims_rz=$(A.dims_rz) and size(b)=$(size(b)) do not match"))
    return reshape(A.matrix \ @view(b[:]), A.dims_rz)
end

## === Broadcasting support for DiscretizedOperator ===


# 1. Make DiscretizedOperator treated as a scalar in broadcasts
# This means A .* B will treat A and B as scalar entities in the broadcast
broadcastable(A::DiscretizedOperator) = A

# # 3. Define how to combine DiscretizedOperator with other broadcast styles
Base.Broadcast.BroadcastStyle(::DOStyle, ::BroadcastStyle) = DOStyle()
Base.Broadcast.BroadcastStyle(::BroadcastStyle, ::DOStyle) = DOStyle()

# Ensure DOStyle wins when mixed with default array style:
combine_styles(::DOStyle, ::BroadcastStyle) = DOStyle()
combine_styles(::BroadcastStyle, ::DOStyle) = DOStyle()

function broadcasted(::DOStyle, f, ops::DiscretizedOperator...)
    mats = map(o->o.matrix, ops)
    bc   = broadcasted(f, mats...)
    M    = materialize(bc)
    return DiscretizedOperator(dims_rz = first(ops).dims_rz, matrix = M)
end

function broadcasted(::DOStyle, f, args...)
    # 1) Identify all operator args
    dops = filter(x -> x isa DiscretizedOperator, args)
    # 2) Extract matrices, leave other args alone
    mats_and_scalars = map(args) do x
        x isa DiscretizedOperator ? x.matrix : x
    end
    # 3) Perform fused broadcast on mixed arguments
    bc = broadcasted(f, mats_and_scalars...)
    # 4) Materialize result to SparseMatrixCSC
    M = materialize(bc)
    # 5) Wrap back into DiscretizedOperator using first operator's dims
    return DiscretizedOperator(dims_rz = first(dops).dims_rz, matrix = M)
end