# The coupled solve on a dense, hot column against the inboard wall of a KSTAR-like domain: the
# grid of the KSTAR field files (R 1.2–2.4 m, Z ±1.2 m) with the KSTAR first wall, whose inboard
# side stands 6 cm (1.5 cells) inside the grid. The column, n = 1e18 m⁻³, Te = 20 eV, radius
# 0.45 m at R = 1.75 m, reaches within 4 cm of that wall, as an inboard-limited start-up plasma
# does. Each step should match the step solved directly.
#
#   julia --project=examples examples/coupled_step/picard_kstar_inboard_limited.jl

include("common.jl")

make() = column(
    "coupled_step/picard_kstar_inboard_limited"; n0 = 1.0e18, Te = 20.0, cenR = 1.75, radius = 0.45,
    manual = kstar_like(0.3), wall_R = KSTAR_WALL_R, wall_Z = KSTAR_WALL_Z,
)
picard_case(make, "picard_kstar_inboard_limited", "KSTAR grid and first wall, inboard-limited column")
