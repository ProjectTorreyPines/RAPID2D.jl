# The coupled solve on a dense, hot column well inside the domain: n = 1e18 m⁻³, Te = 10 eV,
# radius 0.3 m at R = 1.5 m, 0.5 m and more from the edge of the grid. Within a step the
# coupled solve iterates on the boundary flux and the coil currents (common.jl). Here even the
# relaxed iteration (w = 0.5) halves the error at each block solve, and the default (Anderson
# mixing, at most 20 block solves) reproduces the step solved directly, step after step.
#
#   julia --project=examples examples/coupled_step/picard_center_column.jl

include("common.jl")

make() = column("coupled_step/picard_center_column"; n0 = 1.0e18, Te = 10.0)
picard_case(make, "picard_center_column", "dense column, roomy domain")
