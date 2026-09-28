#!/usr/bin/env python3
"""Export absolute KM summaries and 100 paired PSU draws, without SR fits.

Prepare the byte-verified input with scripts/prepare_nhanes_cohort.py first.
The archived coordinates use inclusive entry/death ties and interpolation
between event times, not lifelines' default step quantiles. Missing quantiles
remain missing; each statistic reports its own number of estimable draws.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260924
REPEATS = 100
COHORT_SHA256 = '0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201'
METHOD = 'paired Rao-Wu n-1 PSU bootstrap within wave/stratum; sample SD, ddof=1'


def design_blocks(frame):
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError('Cohort must have a consecutive positional index')
    blocks = []
    for _, block in frame.groupby(['wave', 'SDMVSTRA'], sort=True):
        psus = np.sort(block.SDMVPSU.unique())
        if len(psus) < 2:
            raise ValueError('NHANES stratum has fewer than two PSUs')
        blocks.append([block.index[block.SDMVPSU == psu].to_numpy() for psu in psus])
    return blocks


def bootstrap_weights(blocks, size, replicate):
    """Same draw order and scaling as nhanes/bootstrap_baseline.resampled_frame."""
    if replicate < 1:
        raise ValueError('Bootstrap replicate must be positive')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, replicate, 1]))
    weights = np.zeros(size)
    for block in blocks:
        n = len(block)
        counts = np.bincount(rng.choice(n, n - 1, replace=True), minlength=n)
        for indices, count in zip(block, counts):
            weights[indices] = n * count / (n - 1)
    return weights


def km_metrics(entry, exit_age, event, weights):
    """Product-limit estimate at death times with entry <= t <= exit."""
    entry, exit_age = np.asarray(entry), np.asarray(exit_age)
    event, weights = np.asarray(event, dtype=bool), np.asarray(weights, dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isfinite(entry).all() or not np.isfinite(exit_age).all()
            or (exit_age < entry).any()):
        raise ValueError('Invalid survival intervals or weights')
    positive = weights > 0
    entry, exit_age = entry[positive], exit_age[positive]
    event, weights = event[positive], weights[positive]
    times, inverse = np.unique(exit_age[event], return_inverse=True)
    if not len(times):
        return dict(q75_age=np.nan, median_age=np.nan, q25_age=np.nan, steepness=np.nan)
    deaths = np.bincount(inverse, weights=weights[event])
    entry_order, exit_order = np.argsort(entry), np.argsort(exit_age)
    entered = np.r_[0., np.cumsum(weights[entry_order])]
    exited = np.r_[0., np.cumsum(weights[exit_order])]
    risk = (entered[np.searchsorted(entry[entry_order], times, side='right')]
            - exited[np.searchsorted(exit_age[exit_order], times, side='left')])
    if (risk <= 0).any() or (deaths > risk + 1e-9).any():
        raise ValueError('Invalid weighted risk set')
    survival = np.cumprod(1 - deaths / risk)
    # No endpoint clamping: unresolved quartiles must not become observed ages.
    q75, median, q25 = np.interp(
        [.75, .5, .25], survival[::-1], times[::-1], left=np.nan, right=np.nan)
    steepness = median / (q25 - q75) if q25 > q75 else np.nan
    return dict(q75_age=q75, median_age=median, q25_age=q25, steepness=steepness)


def export(cohort, output):
    if hashlib.sha256(cohort.read_bytes()).hexdigest() != COHORT_SHA256:
        raise ValueError('Cohort differs from the archived likelihood input')
    frame = pd.read_csv(cohort)
    archived_path = ROOT / 'results/nhanes/figure3/exposure_km_points.csv'
    archived = pd.read_csv(archived_path).set_index('group_id')
    groups_path = ROOT / 'analysis/model_fits/nhanes/inputs/groups.json'
    groups = json.loads(groups_path.read_text())
    names = ['all', *archived.index]
    masks = {name: (np.ones(len(frame), dtype=bool) if name == 'all'
                   else frame['group_' + name].eq(1).to_numpy()) for name in names}
    blocks = design_blocks(frame)
    arrays = [frame[c].to_numpy() for c in ('entry', 'exit', 'event')]
    rows, draw_sources = [], []
    for replicate in range(REPEATS + 1):
        weights = (np.ones(len(frame)) if replicate == 0 else
                   bootstrap_weights(blocks, len(frame), replicate))
        if replicate:
            source = ROOT / f'results/nhanes/bootstrap/baseline_precision/rep_{replicate:03d}.json'
            record = json.loads(source.read_text())
            if record['seed'] != SEED or record['replicate'] != replicate:
                raise ValueError(f'Unexpected bootstrap metadata: {source.name}')
            np.testing.assert_allclose(
                [weights.sum(), np.dot(weights, arrays[2])],
                [record['weighted_n'], record['weighted_deaths']], rtol=0, atol=1e-8)
            draw_sources.append(source)
        reference = None
        for name, mask in masks.items():
            metrics = km_metrics(*(a[mask] for a in arrays), weights[mask])
            if name == 'all':
                reference = metrics
            rows.append(dict(replicate=replicate, group_id=name,
                             weighted_n=float(weights[mask].sum()),
                             weighted_deaths=float(np.dot(weights[mask], arrays[2][mask])),
                             **metrics, x=metrics['median_age'] / reference['median_age'],
                             y=metrics['steepness'] / reference['steepness']))
    raw = pd.DataFrame(rows)
    point = raw[raw.replicate == 0].set_index('group_id')
    bootstrap = raw[raw.replicate > 0]
    summaries, comparisons = [], []
    for name in names:
        mask = masks[name]
        sample = bootstrap[bootstrap.group_id == name]
        row = dict(group_id=name, topic='all' if name == 'all' else archived.loc[name, 'topic'],
                   group='All participants' if name == 'all' else groups[name]['label'],
                   n=int(mask.sum()), deaths=int(arrays[2][mask].sum()),
                   bootstrap_repeats=REPEATS, seed=SEED, se_method=METHOD,
                   missing_quantiles='omitted separately per statistic; never endpoint-clamped')
        for metric in ('median_age', 'steepness', 'x', 'y'):
            finite = sample[metric].dropna()
            row[metric] = point.loc[name, metric]
            row[metric + '_se'] = finite.std(ddof=1)
            row[metric + '_valid_repeats'] = len(finite)
            row[metric + '_q025'] = finite.quantile(.025)
            row[metric + '_q975'] = finite.quantile(.975)
        summaries.append(row)
        if name == 'all':
            continue
        if row['n'] != groups[name]['n'] or row['deaths'] != groups[name]['deaths']:
            raise ValueError(f'Group counts differ: {name}')
        mapping = dict(median_age='median_age', steepness='steepness', x='x', y='y',
                       x_err='x_se', y_err='y_se', x_low='x_q025', x_high='x_q975',
                       y_low='y_q025', y_high='y_q975')
        for archived_column, new_column in mapping.items():
            old, new = float(archived.loc[name, archived_column]), float(row[new_column])
            comparisons.append(dict(group_id=name, metric=archived_column, archived=old,
                                    reconstructed=new, difference=new-old,
                                    matches=bool(np.isclose(old, new, rtol=0, atol=1e-9))))
    comparison = pd.DataFrame(comparisons)
    sources = [archived_path, groups_path,
               ROOT / 'analysis/model_fits/nhanes/bootstrap_baseline.py',
               Path(__file__).resolve(), *draw_sources]
    manifest = dict(cohort_sha256=COHORT_SHA256, cohort_n=len(frame),
                    cohort_deaths=int(frame.event.sum()), seed=SEED,
                    seed_sequence='[20260924, replicate, 1]', bootstrap_repeats=REPEATS,
                    bootstrap_method=METHOD, point_estimates='unweighted',
                    km_convention='entry <= death time <= exit; positive-weight death times only; linear inverse interpolation',
                    quantile_convention='unresolved quantiles are NaN; summaries omit missing draws per statistic',
                    table_scope='full cohort plus 23 overlapping exposure groups',
                    archived_normalized_errors_match=bool(comparison[comparison.metric.isin(['x_err', 'y_err'])].matches.all()),
                    all_archived_comparisons_match=bool(comparison.matches.all()),
                    max_absolute_comparison_difference=float(comparison.difference.abs().max()),
                    archived_bootstrap_weight_totals_match=True,
                    numpy_version=np.__version__, pandas_version=pd.__version__,
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / 'supplementaryTable1_source.csv', index=False, float_format='%.17g')
    bootstrap.to_csv(output / 'supplementaryTable1_replicates.csv', index=False, float_format='%.17g')
    comparison.to_csv(output / 'supplementaryTable1_archive_comparison.csv', index=False, float_format='%.17g')
    (output / 'supplementaryTable1_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Exported 24 summaries and {len(bootstrap)} KM-only replicate rows to {output}')
    print(f'All archived comparisons match: {manifest["all_archived_comparisons_match"]}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=ROOT / 'tmp/nhanes_cohort.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/tables')
    args = parser.parse_args()
    manifest = export(args.cohort, args.output)
    if not manifest['all_archived_comparisons_match']:
        raise SystemExit('Archive mismatch: inspect supplementaryTable1_archive_comparison.csv')


if __name__ == '__main__':
    main()
