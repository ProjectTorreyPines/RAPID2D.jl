# The closed-surface flux constraint: its policies, its basis and its state, as types only. They
# sit here, before `types.jl`, so the flags and the run can hold them; the methods are in
# `closed_surface_flux.jl`.

"""
    ClosedSurfaceFlow

How electrons flow along B on closed flux surfaces.
"""
abstract type ClosedSurfaceFlow end

"""
    FreeParallelFlow()

**The default.** The electron parallel velocity is solved on every node alike, and its flux
moves the density on closed surfaces as on open field lines.
"""
struct FreeParallelFlow <: ClosedSurfaceFlow end

"""
    ClosedSurfaceCoupling

How the surface-flux constraint enters the step that solves u∥.
"""
abstract type ClosedSurfaceCoupling end

"""
    BlockConstraint()

**The default.** Above the Ampère gate the surface values of n u∥/B are unknowns of the coupled
u∥–ψ block: the constrained rows and the Galerkin rows join it, so ψ and the coil currents are
solved with the constrained current. Below the gate the constraint is in the u∥ solve.
"""
struct BlockConstraint <: ClosedSurfaceCoupling end

"""
    PostProjection()

Above the Ampère gate the coupled step is solved without the constraint, then u∥ is solved
again with it at the induced field that solve found; the step keeps that ψ, so Ampère's law is
met for the unconstrained current. Kept for comparison with [`BlockConstraint`](@ref). Below
the gate the constraint is in the u∥ solve.
"""
struct PostProjection <: ClosedSurfaceCoupling end

"""
    SurfaceFluxConstraint(; coupling = BlockConstraint())

On the closed region the electron parallel particle flux per B, n u∥ / B, is a surface function,
solved from the momentum equation averaged over each surface, and the density's parallel flux
there is its stream-function flux, which leaves the density unmoved.
"""
struct SurfaceFluxConstraint{C <: ClosedSurfaceCoupling} <: ClosedSurfaceFlow
    coupling::C
end
SurfaceFluxConstraint(; coupling::ClosedSurfaceCoupling = BlockConstraint()) = SurfaceFluxConstraint(coupling)

"""
    ClosedSurfaceBasis{FT}

The constraint domain Ω_c and its surface functions at one ψ:
- `nodes`: the linear ids of Ω_c; `V`: their control volumes 2πR ΔR ΔZ.
- `axis` = (; R, Z, ψ): the O-point. `ψ_lcfs`: the last closed flux surface as the grid
  resolves it, where the surface functions end: the magnetic LCFS, or the last closed surface
  the plasma fills when the plasma ends inside it (nodes at the per-particle density floor are
  not admitted). Nodes of Ω_c beyond it form the edge layer.
- `ψN`, `corner_ψN`: ψN′ = (ψ − ψ_axis)/(ψ_lcfs − ψ_axis) at the nodes and at the dual corners,
  `corner_ψN[i, j]` at (R_i + ΔR/2, Z_j + ΔZ/2).
  `corner_active`: the corners of Ω_c's control volumes inside the LCFS, where H varies; at
  every other corner H takes its value at the LCFS, so no flux appears where ψ elsewhere in the
  plane runs through the region's range.
- `knots`: in ψN′ from the axis (0) to the LCFS (1); a single knot is one constant function.
- `knot_weights` (W, node × knot): w_ks, the weight of knot s at node k, is the piecewise-linear
  basis function of knot s at the node's ψN′, so a node's value is the weighted mean of the two
  knots around it (the weights sum to one). Zero beyond the LCFS, where n u∥/B ends.
  `knot_weights_held`: the same, held at their LCFS values beyond it, for temperatures, which do
  not end there.

Build with [`closed_surface_basis`](@ref).
"""
struct ClosedSurfaceBasis{FT <: AbstractFloat}
    nodes::Vector{Int}
    V::Vector{FT}
    axis::@NamedTuple{R::FT, Z::FT, ψ::FT}
    ψ_lcfs::FT
    ψN::Vector{FT}
    corner_ψN::Matrix{FT}
    corner_active::BitMatrix
    knots::Vector{FT}
    knot_weights::SparseMatrixCSC{FT, Int}
    knot_weights_held::SparseMatrixCSC{FT, Int}
    dims::Tuple{Int, Int}
end

"""
    ConstrainedSystem{FT}

A square sparse system S x = r whose first unknowns are u∥ on the grid's nodes, with the
surface-flux constraint on the nodes `cu` inside the LCFS: with y = c (n u∥/B) at the knots,
c = `scale`, their rows become u_k − (T y)_k = 0, T = diag(B/(n c)) W, and M Galerkin rows
Q (S x − r) = extra, Q = Wᵀ diag(V B) normalized by `qs`, are appended; the unknowns are [x; y].
W = `knot_weights` at `cu`. The pattern, of S and of the knot weights at `cu`, is built once; [`update_constrained_system!`](@ref) writes values and refactorizes with the LU's
symbolic analysis kept. [`constrained_system`](@ref) gives a new one when the pattern changes.
"""
mutable struct ConstrainedSystem{FT <: AbstractFloat}
    key::UInt                       # hash of S's pattern, cu and the knot weights' pattern at cu
    L::Int                          # S's size
    cu::Vector{Int}
    matrix::SparseMatrixCSC{FT, Int}  # (L + M) × (L + M)
    pos_S::Vector{Int}              # each stored entry of S: its position, 0 on a constrained row
    pos_I::Vector{Int}              # (k, k) of each constrained row
    basis_row::Vector{Int}          # each stored knot weight at cu (W's CSC order): its row in cu
    basis_col::Vector{Int}          # and its knot
    pos_T::Vector{Int}              # (cu[c], L + s) of each
    pos_shift::Vector{Int}          # (L + s, cu[c]) of each
    gal_S::Vector{Int}              # the Galerkin rows' products: a stored entry of S,
    gal_basis::Vector{Int}          # a knot weight in its row,
    gal_pos::Vector{Int}            # and the position (L + s, column) they add to
    q::Vector{FT}                   # the Galerkin row coefficient of each knot weight, W V B / qs
    qs::Vector{FT}                  # each Galerkin row's norm
    scale::FT                       # the mean B/n: y = scale × (n u∥/B) is a velocity
    solver::SparseLUSolver{FT}
    fresh::Bool                     # the pattern has not been factorized yet
end

"""
    SurfaceFluxSolve{FT}(NR, NZ)

What the constraint's last u∥ solve leaves for the rest of the step: `Γe_para_over_B`, the
electron parallel particle flux over B, n u∥/B, a surface function, at the knots of the basis with `signature` (empty: none yet on this
basis), its stream-function face fluxes `ΓR`, `ΓZ` laid out as in [`stream_function_fluxes`](@ref),
and the `compression` B_pᵀ(n u∥/B) on the grid (the pressure operator's adjoint; zero off its domain).
"""
mutable struct SurfaceFluxSolve{FT <: AbstractFloat}
    Γe_para_over_B::Vector{FT}
    signature::UInt
    ΓR::Matrix{FT}
    ΓZ::Matrix{FT}
    compression::Matrix{FT}
end
SurfaceFluxSolve{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat} =
    SurfaceFluxSolve{FT}(FT[], UInt(0), zeros(FT, NR - 1, NZ), zeros(FT, NR, NZ - 1), zeros(FT, NR, NZ))

"""
    SurfaceFluxLedger{FT}()

The constraint's cumulative signed energy ledger [J]: the kinetic energy it removed on its first
step (`activation`), on steps where the basis changed (`basis_change`) and on other steps
(`projection`), and the change from remapping u∥ to the step's new density (`remap`).
"""
mutable struct SurfaceFluxLedger{FT <: AbstractFloat}
    activation::FT
    basis_change::FT
    projection::FT
    remap::FT
end
SurfaceFluxLedger{FT}() where {FT <: AbstractFloat} = SurfaceFluxLedger{FT}(zero(FT), zero(FT), zero(FT), zero(FT))

"""
    ClosedSurfaceState{FT}(NR, NZ)

The constraint in a run, by lifetime: the `basis` of the step (`nothing` when there is none),
rebuilt after each field-line analysis; the constrained systems ([`ConstrainedSystem`](@ref)) of
the u∥ step (`u_system`) and of the coupled u∥–ψ block (`block_system`), each kept while its
pattern holds; the `solve` of the step ([`SurfaceFluxSolve`](@ref)); and the cumulative `ledger`
([`SurfaceFluxLedger`](@ref)). A step solved again above the Ampère gate puts `solve` and
`ledger` back as they were.
"""
mutable struct ClosedSurfaceState{FT <: AbstractFloat}
    basis::Union{Nothing, ClosedSurfaceBasis{FT}}
    u_system::Union{Nothing, ConstrainedSystem{FT}}
    block_system::Union{Nothing, ConstrainedSystem{FT}}
    solve::SurfaceFluxSolve{FT}
    ledger::SurfaceFluxLedger{FT}
end
ClosedSurfaceState{FT}(NR::Int, NZ::Int) where {FT <: AbstractFloat} =
    ClosedSurfaceState{FT}(nothing, nothing, nothing, SurfaceFluxSolve{FT}(NR, NZ), SurfaceFluxLedger{FT}())
