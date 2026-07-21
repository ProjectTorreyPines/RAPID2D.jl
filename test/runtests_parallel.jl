# runtests_parallel.jl — run the RAPID2D @testitem suite in PARALLEL via ReTestItems,
# leaving test/ and the TestItemRunner path (test/runtests.jl, VS Code) untouched.
#
# ── Why a shadow copy is needed ──────────────────────────────────────────────────────
# TestItemRunner and ReTestItems agree on `@testitem` and on NOTHING else: the former
# understands `@testsnippet` but not `@testsetup`, the latter the reverse. The source uses
# `@testsnippet` because its body is SPLICED into each testitem, so fixtures land in a flat
# name scope and every item gets a fresh copy. So this script builds a transient shadow
# tree rewriting each `@testsnippet Name begin … end` into `@testsetup module Name … end`.
#
# ReTestItems also rejects ANY top-level expression that is not @testitem/@testsetup, and
# `@testsnippet` is itself such a stray — so a file with a CO-LOCATED snippet is copied
# with the snippet ranges blanked to spaces (newlines preserved so line numbers line up),
# while files whose snippets live in a separate setup_*.jl are symlinked untouched. Test
# files are already named `*_test.jl`, exactly what ReTestItems discovers, so nothing has
# to be renamed.
#
# Corollary: the setup_*.jl files must NOT be named `*_testsetup.jl`. That suffix would
# make ReTestItems load them directly, where `@testsnippet` is invalid.
#
# ── Usage ────────────────────────────────────────────────────────────────────────────
# Canonical (through Pkg.test — test/runtests.jl routes here when NWORKERS > 0):
#     RETESTITEMS_NWORKERS=4 cc-julia-test-runner .
#     RETESTITEMS_NWORKERS=4 cc-julia-test-runner . physics
# Standalone (skips the Pkg.test sandbox; test env must already be instantiated):
#     julia --project=test test/runtests_parallel.jl physics --nworkers 4
#
# NOTE: parallelism pays off on LARGE runs. Wall-clock is bounded below by the slowest item
# ("RAPID coils evolution without plasma" ~1m44s, plasma-coil coupling ~1m05s); on a small
# keyword subset plain `cc-julia-test-runner .` is usually faster.

using ReTestItems

const PKGROOT = normpath(joinpath(@__DIR__, ".."))
const REALTEST = joinpath(PKGROOT, "test")
# In-repo (git-ignored) and PID-suffixed: with a fixed name, two concurrent runs in the
# same checkout would delete and rebuild each other's tree mid-flight.
const SHADOW = joinpath(PKGROOT, "_partest_shadow_$(getpid())")
const STANDALONE = !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == abspath(@__FILE__)

# Directories under test/ NOT walked for test files. `data` is deliberately NOT mirrored
# into the shadow: fixtures must be addressed via `pkgdir(RAPID2D)`, which resolves to the
# real package root no matter where the shadowed code ends up being evaluated.
const SKIP_DIRS = ("tmp_regression", "output", "data", basename(SHADOW))

# ── args ─────────────────────────────────────────────────────────────────────────────
# In a function so assignments are not trapped in a hard local scope at top level.
function parse_args(args)
    patterns = String[]
    nworkers = nothing
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--nworkers"
            nworkers = parse(Int, args[i + 1]); i += 2
        elseif startswith(a, "--nworkers=")
            nworkers = parse(Int, split(a, '=')[2]); i += 1
        elseif isempty(a) || a == "."       # cc-julia-test-runner muscle memory
            i += 1
        else
            push!(patterns, a); i += 1
        end
    end
    return patterns, nworkers
end

# ── scan a file for @testsnippet blocks ──────────────────────────────────────────────
# Returns (src, snippets, cuts, item_names): `snippets` are (name, body) pairs to re-emit
# as @testsetup modules, `cuts` the char ranges of the @testsnippet blocks themselves, to
# blank from the shadow copy. Walks top-level expressions with Meta.parse rather than
# regex so nested begin/end inside a snippet body cannot confuse the extent.
function scan_snippets(path)
    src = read(path, String)
    snips = Tuple{String, String}[]
    cuts = Tuple{Int, Int}[]
    item_names = String[]
    pos = firstindex(src)
    while true
        ex, np = Meta.parse(src, pos; raise = false)
        (ex === nothing || np <= pos) && break
        stop = prevind(src, np)
        if ex isa Expr && ex.head === :macrocall
            mname = ex.args[1]
            if mname === Symbol("@testsnippet") && length(ex.args) >= 4
                # Match begin/end as WHOLE WORDS: on some Julia versions Meta.parse folds a
                # trailing comment into this expression's span, and a plain findlast("end", …)
                # would latch onto the "end" inside a word like "depend" and truncate the body.
                chunk = src[pos:stop]
                bs = collect(eachmatch(r"\bbegin\b", chunk))
                es = collect(eachmatch(r"\bend\b", chunk))
                if !isempty(bs) && !isempty(es)
                    name = string(ex.args[3])
                    bstop = bs[1].offset + ncodeunits(bs[1].match)
                    estart = es[end].offset
                    body = strip(chunk[bstop:prevind(chunk, estart)])
                    push!(snips, (name, String(body)))
                end
                push!(cuts, (pos, stop))   # blank the @testsnippet from the shadow copy
            elseif mname === Symbol("@testitem") && length(ex.args) >= 3 && ex.args[3] isa String
                push!(item_names, ex.args[3])
            end
        end
        np > lastindex(src) && break
        pos = np
    end
    return src, snips, cuts, item_names
end

# Overwrite every `cuts` range with spaces but keep newlines, so the shadow copy's LINE
# numbers still match the real file and stack traces stay meaningful.
#
# `pairs(src)` yields BYTE indices, the same index space Meta.parse reports. A collect()ed
# Char vector would be wrong: these files are full of Unicode identifiers (∇𝐃∇, ψ, θ,
# ν_en_iz), so char and byte indices diverge and the blanking would corrupt the source.
function blank_ranges(src, cuts)
    isempty(cuts) && return src
    io = IOBuffer()
    for (i, c) in pairs(src)
        if c != '\n' && any(((a, b),) -> a <= i <= b, cuts)
            write(io, ' ')
        else
            write(io, c)
        end
    end
    return String(take!(io))
end

# A @testsetup module needs its OWN imports — unlike @testsnippet, which is inlined into a
# testitem that already had `using RAPID2D` injected. Add that, then auto-export every
# binding so `using .Name` exposes them unqualified, reproducing @testsnippet's flat scope.
function to_testsetup(name, body)
    return """
    @testsetup module $name
        using Test
        using RAPID2D
    $body
        for var"#en" in names(@__MODULE__; all = true)
            startswith(string(var"#en"), "#") && continue
            var"#en" in (:eval, :include) && continue
            isdefined(@__MODULE__, var"#en") || continue
            @eval export \$var"#en"
        end
    end
    """
end

# ── shadow tree ──────────────────────────────────────────────────────────────────────
function with_shadow(f, dir)
    rm(dir; force = true, recursive = true)
    mkpath(dir)
    try
        return f(dir)
    finally
        rm(dir; force = true, recursive = true)
    end
end

_matches(patterns, s) = isempty(patterns) || any(p -> occursin(lowercase(p), lowercase(s)), patterns)

function build_and_run(shadow, patterns, nworkers)
    setup_defs = String[]
    selected = String[]

    for (root, dirs, files) in walkdir(REALTEST)
        filter!(d -> !(d in SKIP_DIRS), dirs)     # prune, do not descend
        for file in sort(files)
            endswith(file, ".jl") || continue
            real = joinpath(root, file)
            src, snips, cuts, item_names = scan_snippets(real)

            # Collect every @testsnippet in the tree, wherever it lives.
            for (nm, body) in snips
                push!(setup_defs, to_testsetup(nm, body))
            end

            # Only *_test.jl files carry testitems worth selecting.
            endswith(file, "_test.jl") || continue
            isempty(item_names) && continue
            file_hit = _matches(patterns, file)
            any_item_hit = any(n -> _matches(patterns, n), item_names)
            (file_hit || any_item_hit) || continue

            rel = relpath(real, REALTEST)
            dst = joinpath(shadow, rel)
            mkpath(dirname(dst))
            # No co-located @testsnippet → symlink verbatim. Otherwise copy with the
            # snippets blanked; they were captured above and re-emitted in the wrapper.
            if isempty(cuts)
                symlink(real, dst)
            else
                write(dst, blank_ranges(src, cuts))
            end
            push!(selected, dst)
        end
    end

    # ReTestItems discovers setup modules only in *_testsetup.jl files.
    setupfile = joinpath(shadow, "aa_wrapper_testsetup.jl")
    write(setupfile, join(setup_defs, "\n\n"))

    isempty(selected) && (println("── partest: no test files matched $(patterns) ──"); return)

    # Set :SOURCE_PATH to the real runtests.jl so ReTestItems takes its "running under
    # Pkg.test" branch (skips TestEnv.activate and uses the already-active environment).
    task_local_storage(:SOURCE_PATH, joinpath(REALTEST, "runtests.jl"))

    sel = isempty(patterns) ? "all" : join(repr.(patterns), ", ")
    println("── partest ─ files=$(length(selected))  nworkers=$nworkers  patterns: $sel ──")
    flush(stdout)

    # Tag gating, kept in lockstep with test/runtests.jl's serial filter. ReTestItems'
    # own `tags=` kwarg is an INCLUSION filter, so exclusion has to go through the
    # shouldrun function (it receives a TestItemMetadata carrying `.tags`).
    want_regression = get(ENV, "RAPID_RUN_REGRESSION", "false") == "true"
    want_broken = get(ENV, "RAPID_RUN_BROKEN", "false") == "true"
    ti_filter = ti -> begin
        isempty(patterns) || return true      # explicit request wins, as in runtests.jl
        :broken in ti.tags && return want_broken
        :regression in ti.tags && return want_regression
        return true
    end

    t = @elapsed runtests(
        ti_filter, setupfile, selected...;
        nworkers = nworkers,
        nworker_threads = 1,
        retries = parse(Int, get(ENV, "RETESTITEMS_RETRIES", "0")),
        verbose_results = false,
        report = false,
        logs = :issues,
    )
    println("\n>>> partest done: $(length(selected)) files · nworkers=$nworkers · $(round(t, digits = 2)) s")
    return nothing
end

function main()
    patterns, nworkers_cli = parse_args(ARGS)
    nworkers = something(
        nworkers_cli,
        STANDALONE ? 2 : parse(Int, get(ENV, "RETESTITEMS_NWORKERS", "2")),
    )
    return with_shadow(SHADOW) do shadow
        build_and_run(shadow, patterns, nworkers)
    end
end

main()
