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
