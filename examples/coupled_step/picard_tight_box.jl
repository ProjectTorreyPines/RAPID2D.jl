# The coupled solve on a dense, hot column that fills a box wall standing one cell inside the
# grid (R 1.0–2.2 m, Z ±0.8 m): n = 1e18 m⁻³, Te = 10 eV, radius 0.55 m at R = 1.6 m. The
# column's own flux returns through the boundary values, which the iteration holds fixed for each
# solve inside the domain; so close to the grid edge that return overshoots. Each step should
# match the step solved directly.
#
#   julia --project=examples examples/coupled_step/picard_tight_box.jl

include("common.jl")

make() = column(
    "coupled_step/picard_tight_box"; n0 = 1.0e18, Te = 10.0, cenR = 1.6, radius = 0.55, manual = tight_box(0.3),
)
picard_case(make, "picard_tight_box", "column filling a box one cell inside the grid")
