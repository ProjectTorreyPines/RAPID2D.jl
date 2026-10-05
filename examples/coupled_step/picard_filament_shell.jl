# The coupled solve on a dense, hot column inside a passive conducting shell: 24 copper
# filaments on a circle of radius 0.35 m around a column of radius 0.3 m at R = 1.5 m
# (n = 1e18 m⁻³, Te = 10 eV), inside the grid, as a vessel wall modelled by filaments would be.
# The filaments carry the eddy currents that hold the plasma's flux; each step should match the
# step iterated to convergence.
#
#   julia --project=examples examples/coupled_step/picard_filament_shell.jl

include("common.jl")

make() = filament_shell!(column("coupled_step/picard_filament_shell"; n0 = 1.0e18, Te = 10.0); cenR = 1.5, r = 0.35)
picard_case(make, "picard_filament_shell", "column inside a shell of 24 copper filaments")
