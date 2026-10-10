# How a field line is followed in the poloidal plane: the tracer policies. Their methods are in
# `field_line_tracing.jl`; the types sit here, before `types.jl`, so the simulation flags can
# hold one. internal/docs/src/notes/plans/PLAN_flf-contours.md §6.

"""
    FieldLineTracer

How a field line is followed in the (R, Z) plane. In axisymmetry its poloidal projection is a
level set of ψ, so each tracer follows a ψ contour through its own representation of the
field. The analyses built on tracing (field-line lengths, flux-surface and field-line
averages) take a tracer and never learn which one it is.
"""
abstract type FieldLineTracer end

"""
    MarchingSquaresTracer()

The contour of the bilinear ψ on the grid, followed cell by cell: each step crosses one cell,
from the edge where the contour enters to the edge where it leaves, and a saddle cell connects
its edges by the value at its centre. A trace stops on the edge of the first cell outside the
wall. B_pol at each vertex is |∇ψ|/R of the bicubic ψ. The fast option.
"""
struct MarchingSquaresTracer <: FieldLineTracer end

"""
    CubicTracer(; max_turn = deg2rad(5), max_weight_change = 0.02, max_step_cells = 0.5, max_steps = 20_000)

The level set of the bicubic ψ, followed by predictor–corrector continuation: a step along the
tangent, then Newton back onto the level. A step is halved until the tangent turns by at most
`max_turn` and the integrand R/|∇ψ| changes by at most a fraction `max_weight_change` (which
keeps the path resolved toward an X-point, where |∇ψ| → 0), and is never longer than
`max_step_cells` grid spacings. The wall is found where a step's chord first enters a cell
outside it. The path is not limited by the cells, so a contour a few cells across, or one
beside an X-point, keeps its length. The accurate option; it follows IMAS.jl
`fluxsurfaces_cubic.jl`.
"""
struct CubicTracer{FT <: AbstractFloat} <: FieldLineTracer
    max_turn::FT
    max_weight_change::FT
    max_step_cells::FT
    max_steps::Int
    function CubicTracer(;
            max_turn::Real = deg2rad(5.0), max_weight_change::Real = 0.02,
            max_step_cells::Real = 0.5, max_steps::Int = 20_000,
        )
        0 < max_turn < π / 2 || throw(ArgumentError("max_turn must be in (0, π/2), got $max_turn"))
        0 < max_weight_change < 1 || throw(ArgumentError("max_weight_change must be in (0, 1), got $max_weight_change"))
        max_step_cells > 0 || throw(ArgumentError("max_step_cells must be positive, got $max_step_cells"))
        max_steps > 0 || throw(ArgumentError("max_steps must be positive, got $max_steps"))
        FT = float(promote_type(typeof(max_turn), typeof(max_weight_change), typeof(max_step_cells)))
        return new{FT}(FT(max_turn), FT(max_weight_change), FT(max_step_cells), max_steps)
    end
end

"""
    RK4Tracer(; step_cells = 0.5)

Classical RK4 along the unit poloidal field of the bicubic-interpolated nodal B_R, B_Z, in
fixed steps of `step_cells` grid spacings. It follows B, not the level, so it drifts off the
ψ contour, and it checks the wall only where a step ends. A trace has closed when its
direction has turned through 2π. The field-line following RAPID2D used first, kept as the
reference: lengths from it are right-endpoint sums, as they always were.
"""
struct RK4Tracer{FT <: AbstractFloat} <: FieldLineTracer
    step_cells::FT
    function RK4Tracer(; step_cells::Real = 0.5)
        step_cells > 0 || throw(ArgumentError("step_cells must be positive, got $step_cells"))
        return new{float(typeof(step_cells))}(step_cells)
    end
end

# whether a tracer reads the nodal B_R, B_Z rather than ψ
needs_nodal_B(::FieldLineTracer) = false
needs_nodal_B(::RK4Tracer) = true
