# The figures of a finished run from its output folder alone, with no simulation and no input
# files: the grid and its wall masks (grid.bp), the wall (wall.bp), R0B0 (config.bp) and the
# snapshots (snap0D.bp, snap2D.bp). Into the same folder: the five-panel movie (log nₑ, nₑ, Tₑ, u∥,
# Jϕ, each over its colour bar) and snapshot grid, with the basis's and the magnetic LCFS, the O- and
# X-points, the current centroid and the LCFS's geometric centre as the snapshots recorded them; the
# flux-surface-averaged profiles on the closed surfaces with their Grad–Shafranov terms, at the last
# snapshot and over time; the column table and figure; the 0D traces and dashboard. With
# `--profiles`, only the profiles. With `--halpha`, only the Hα emission of e + H₂ (the model's
# only Hα channel): its movie (log10 ε_Hα, ε_Hα, n_H₂), halpha.mp4, and the vessel's photon rate
# against time, halpha_trace.png. With `--presentation`, only the movie for slides (the time and
# the panel titles larger, no run title or legend): fields_for_presentation.mp4, or with
# `--halpha` halpha_for_presentation.mp4.
#
#   julia --project=examples examples/plot_run.jl <output folder> [title]
#   julia --project=examples examples/plot_run.jl <output folder> --profiles [title]
#   julia --project=examples examples/plot_run.jl <output folder> --halpha [--presentation] [title]
#   julia --project=examples examples/plot_run.jl <output folder> --presentation

include("common.jl")
flags = filter(startswith("--"), ARGS)
args = filter(!startswith("--"), ARGS)
isempty(args) && error("usage: julia --project=examples examples/plot_run.jl <output folder> [title] [--profiles | --halpha | --presentation]")
dir = args[1]
title = length(args) >= 2 ? args[2] : basename(normpath(dir))
if "--halpha" in flags
    run = load_run(dir)
    presentation = "--presentation" in flags
    plot_halpha_trace(run; file = joinpath(dir, "halpha_trace.png"), title)
    file = joinpath(dir, presentation ? "halpha_for_presentation.mp4" : "halpha.mp4")
    println("movie in ", animate_halpha(run; file, title, presentation))
elseif "--presentation" in flags
    println("movie in ", animate_fields(load_run(dir); file = joinpath(dir, "fields_for_presentation.mp4"), title, presentation = true))
elseif "--profiles" in flags
    plot_run_profiles(load_run(dir); title)
    println("profiles in ", dir)
else
    plot_run_figures(load_run(dir); title)
end
