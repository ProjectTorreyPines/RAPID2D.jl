# The figures of a finished run from its output folder alone, with no simulation and no input
# files: the grid and its wall masks (grid.bp), the wall (wall.bp) and the snapshots (snap0D.bp,
# snap2D.bp). Into the same folder: the five-panel movie (log nₑ, nₑ, Tₑ, u∥, Jϕ, each over its
# colour bar) and snapshot grid, with the basis's and the magnetic LCFS, the O- and X-points, the
# current centroid and the LCFS's geometric centre as the snapshots recorded them; the column
# table and figure; the 0D traces and dashboard.
#
#   julia --project=examples examples/plot_run.jl <output folder> [title]

include("common.jl")
isempty(ARGS) && error("usage: julia --project=examples examples/plot_run.jl <output folder> [title]")
dir = ARGS[1]
title = length(ARGS) >= 2 ? ARGS[2] : basename(normpath(dir))
plot_run_figures(load_run(dir); title)
