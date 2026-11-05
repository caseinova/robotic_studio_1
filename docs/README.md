# Project page

This static site maps the integrated-project brief to repository evidence. It intentionally labels missing results and submission details rather than inventing evidence.

Preview from the repository root:

```sh
python -m http.server 8000 --directory docs
```

Open http://localhost:8000. To publish with GitHub Pages, push this branch, then select **Settings → Pages → Deploy from a branch → project_page → /docs**. No dependencies or build step are needed. If deploying from another branch, update `sourceRoot` in `page.js` to that branch. This setup does not publish or change repository settings automatically.

The JSON files in `docs/data/` are snapshots of the active configuration. Refresh them from `config/controller_config.json` and `config/estimator_config.json` whenever updating the page. Static prose also records noise and MPC defaults; review it after configuration changes. Run-specific evidence must record its own exact configuration and commit.

Before submission, replace the explicitly marked gaps: team/group, video, calibration measurements, model analysis, parameter rationale, repeated randomised/disturbed trial results, figures, individual contributions and acknowledgements. Confirm the fixed-camera departure from the wrist-camera brief with the teaching team.

The page is based on `Sensors_Control_Integrated_Project v2.pdf` supplied alongside the checkout, the repository Markdown guides and checked-in controller/estimator configuration and source. It is a project-page scaffold, not a completed assessment report.
