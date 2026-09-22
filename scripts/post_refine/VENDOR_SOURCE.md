# Post-refine vendor source

The files under `vendor/fysiverse_3d/` are the minimal source subset used by
the FysicsMagic inference-side post-refine runner. The runner is enabled by
default, but can be disabled with `--skip-refine`. This public copy is
self-contained and does not require a private source path.

The local runner also contains the small case-preparation and metric helpers
under `scripts/`. The runtime environment is `fysiverse-refine` by default and
the Blender executable is selected by `POST_REFINE_BLENDER` (or `blender` on
`PATH`).

By default, `post_refine/run.sh` uses this local vendor copy. Setting
`POST_REFINE_PROJECT_ROOT` or `POST_REFINE_RUNNER` explicitly enables the
legacy external-runner compatibility path.

The subset was adapted from the FysicsMagic inference-side refinement
implementation. The project owner has confirmed that Fysics AI contributors
may distribute this included source under the repository's Apache License 2.0.
See the top-level `LICENSE` and `NOTICE` files. This grant applies only to the
included source and does not replace the licenses of Blender, nvdiffrast, or
other separately installed runtime dependencies.
