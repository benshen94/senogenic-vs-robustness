# Optional senogenic-heterogeneity profile computation

Plotting Extended Data Fig. 1 reads `results/senogenic_heterogeneity/` and never
calls these workers. Each numerical worker requires `--recompute`. Outputs go
to `tmp/senogenic_refit/`, not the frozen records.

For selective numerical checks of archived candidates, first run
`python3 analysis/model_fits/senogenic_heterogeneity/prepare_saved.py`. For a
fresh full profile, omit this preparation step and use an empty rerun directory.

Run from the repository root in this order, waiting for each stage to finish:

1. `LSB_JOBINDEX=1 python3 analysis/model_fits/senogenic_heterogeneity/fit.py --recompute`
   selects one of nine profiles. Indices 1--9 correspond to imposed timescale
   CVs 0, .025, .04, .05, .075, .10, .15, .20, .25. Complete all nine.
2. `python3 analysis/model_fits/senogenic_heterogeneity/refine_zero.py --recompute`
   refines the zero-spread profile at production resolution.
3. `LSB_JOBINDEX=1 python3 analysis/model_fits/senogenic_heterogeneity/stress.py --recompute`
   checks wider bounds. Indices 1 and 2 select 15% and 25% spread, respectively.
4. `python3 analysis/model_fits/senogenic_heterogeneity/refine_four.py --recompute`
   performs the wider-bound 4% diagnostic. It is retained separately and is
   deliberately excluded from the common-bound line.
5. `LSB_JOBINDEX=1 python3 analysis/model_fits/senogenic_heterogeneity/rank_candidates.py --recompute`
   reevaluates stage/anchor/neighbor candidates on the production grid. Complete
   indices 1--9 after the profile and zero-refinement stages. The 7.5% selection
   receives an additional fine-grid evaluation.
6. `LSB_JOBINDEX=1 python3 analysis/model_fits/senogenic_heterogeneity/check.py --recompute`
   compares production and fine grids for the profile and anchor. Archived
   checks use indices 1, 3 and 7 (0%, 4%, 15%).

The original manifest records WEXAC job 838366, nine concurrent tasks, two
slots per task and a 45-minute allocation in the short queue. Additional job
IDs, failed attempts and retries remain in the manifest. Those are provenance,
not tasks submitted by this repository. Set `SR_THREADS` to the allocated
thread count (default one). The numerical levels and optimizer bounds are
explicit in `model.py` and `fit.py`; finer checks can be substantially slower.

These are path-adapted copies of the original optimization and selection logic.
Manifest hashes refer to the original scripts. Local imports were substituted
for the private historical pipeline's data/save helpers. The bundled Sweden
2019 age-specific deaths/exposures and baseline are unchanged.

No profile bootstrap was performed. Common-bound selected scores are minima
among evaluated candidates, not certified global minima. Wider-bound results
must not silently replace points in the common-bound figure. For interpretation
and the timescale-distribution construction, see
`docs/figure_methods/senogenic_heterogeneity_fp.md`.
