# Project page

This static site describes the project's implementation using repository source,
configuration snapshots and installation details reported by the team. It labels
unpublished results rather than treating local experiment files as public evidence.

Preview from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. To publish with GitHub Pages, push this branch, then select **Settings → Pages → Deploy from a branch → project_page → /docs**. No dependencies or build step are needed. If deploying from another branch, update `sourceRoot` in `page.js` to that branch. This setup does not publish or change repository settings automatically.

The JSON files in `docs/data/` are snapshots of the active configuration. Refresh them from `config/controller_config.json` and `config/estimator_config.json` whenever updating the page. Static prose also records noise and MPC defaults; review it after configuration changes. Run-specific evidence must record its own exact configuration and commit.

The published simulation evidence lives in `docs/results/`, including original
plots, CSVs, a provenance manifest and available configuration snapshots. See
[the archive notes](results/README.md) for metric definitions and reproduction.
Run `python docs/analyze_results.py` from the repository root to regenerate the
summary JSON, RMSE comparison and XY trajectory plot. Review the static results table and prose if
the archive changes; they are not automatically regenerated.

Only cite material readers can access through the published site or linked
repository. Do not cite ignored `experiment_runs/` or `simulation_plots/` files,
local assessment documents, or unpublished logs. Before adding a numerical
result, publish its data, exact configuration and source commit together, and
distinguish controller-feedback errors from independently measured truth errors.
Check every `data-src` target is tracked, and every relative asset is included
in the deployed `docs/` directory. Commit and push the page, JSON snapshots,
math scripts and bundled KaTeX assets together before publishing.

Equations use LaTeX rendered by the locally bundled KaTeX 0.16.22 in
`vendor/katex/`. Visitors do not need to install anything, and rendering does
not require a CDN. Keep its stylesheet, JavaScript, fonts and license together.
Write display equations inside `<div class="equation">` and inline expressions
inside `<span class="math-inline">`, without dollar-sign delimiters. Escape
HTML characters in the source: use `&amp;` for LaTeX alignment separators and
`&lt;` for less-than signs. `math.js` renders these elements and leaves code
blocks untouched. Wide equations scroll horizontally, including with keyboard
focus. KaTeX also supplies MathML for assistive technology. Browser-console
errors or elements with the `math-error` class indicate invalid expressions.

The overview links to the notation table at the bottom of the page. Keep that
table synchronized when changing equations, including units and any symbols
with context-dependent meanings. Introduce each component through its purpose,
design trade-off and assumptions before describing its implementation.

The current results cover four low-noise layouts, their high-noise counterparts,
and the latest four high-noise EKF-off trials with matching initial conditions.
The EKF-off snapshots now match the high-noise EKF-on settings, including the
distance-dependent depth-noise coefficient. The `new_dest_1` and
`new_dest_2` recordings are video-only and excluded from quantitative analysis.
The manifest, rather than a directory glob, selects which archives are analyzed.
Superseded archives have been removed from `docs/results/`; only the twelve
selected trial directories and shared analysis files remain.
Public calibration
validation, controlled repeated trials and complete source-commit provenance
remain necessary for stronger accuracy or reliability claims.
The project demonstration video is embedded from `project_video.mp4`.
The authorship section records the sole author, group details, contribution and AI assistance.
The PBVS/MPC table compares implementations, not measured performance.
