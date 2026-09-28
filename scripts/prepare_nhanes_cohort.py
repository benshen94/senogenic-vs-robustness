#!/usr/bin/env python3
"""Reconstruct the likelihood cohort from bundled public-use NHANES inputs."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ageing_packages.hetero_analysis.nhanes_analysis import (
    TOPIC_CONFIGS, _apply_grouping_strategy, get_topic_df, load_core,
)

TOPICS = {
    'diet': ['Poor', 'Good'],
    'number_of_friends': ['0 friends', '1+ friends'],
    'income': ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
    'alcohol': ['>4 drinks/day', '0-1 drink/day'],
    'physical_activity': ['No Activity', 'Some Activity'],
    'sleep_duration': ['1-<5 hours', '5-<7 hours', '7-<9 hours', '>=9 hours'],
    'sleep_frailty': ['Q4 (highest)', 'Q1 (lowest)'],
    'church_frequency': ['never', 'sometimes', 'weekly'],
    'education_level': ['no highschool', 'some college'],
}
EXPECTED_SHA256 = '0fb93c36eb9ee48b1a5d5d1aab02f4e48444a4530b0ae03ba2740619cc9b0201'


def clean(frame):
    frame = frame.copy()
    cutoff = np.where(frame.year.str[:4].astype(int) >= 2007, 80, 85)
    frame = frame.loc[(frame.entry_age < cutoff) & (frame.exit_age > 20)
                      & frame.event.isin([0, 1])].dropna(
                          subset=['entry_age', 'exit_age', 'event'])
    frame['entry_age'] = frame.entry_age.clip(lower=20)
    if (frame.exit_age < frame.entry_age).any() or frame.SEQN.duplicated().any():
        raise ValueError('Invalid observation interval or duplicate participant')
    same = frame.exit_age == frame.entry_age
    frame.loc[same, 'exit_age'] = frame.loc[same, 'entry_age'] + 1 / 24
    return frame


def build(source):
    baseline = clean(load_core(source))
    output = baseline[['SEQN', 'wave', 'entry_age', 'exit_age', 'event']].rename(
        columns={'entry_age': 'entry', 'exit_age': 'exit'})
    expected = json.loads((ROOT / 'analysis/model_fits/nhanes/inputs/groups.json').read_text())
    for topic, labels in TOPICS.items():
        raw = get_topic_df(topic, source)
        if topic == 'sleep_duration':
            raw = raw.loc[~raw.sleep_hours.isin([77, 99])].copy()
        frame, column = _apply_grouping_strategy(raw, TOPIC_CONFIGS[topic])
        frame = frame.copy()
        frame['label'] = frame[column].astype(str).str.replace('\u2265', '>=', regex=False)
        frame = clean(frame.loc[frame.label.isin(labels)])
        for index, label in enumerate(labels):
            key = f'{topic}__{index}'
            subset = frame.loc[frame.label == label]
            if len(subset) != expected[key]['n'] or int(subset.event.sum()) != expected[key]['deaths']:
                raise ValueError(f'Archived group counts do not match: {key}')
            output['group_' + key] = baseline.SEQN.isin(subset.SEQN).to_numpy().astype(int)
    demographic = pd.concat([
        pd.read_sas(path, format='xport')[['SEQN', 'SDMVSTRA', 'SDMVPSU']]
        for path in sorted((source / 'demo').glob('*.xpt'))])
    output = output.merge(demographic, on='SEQN', how='left', validate='one_to_one')
    if output[['SDMVSTRA', 'SDMVPSU']].isna().any().any():
        raise ValueError('Missing survey design identifiers')
    if len(output) != 55800 or int(output.event.sum()) != 7260:
        raise ValueError('Baseline counts do not match archived fits')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/nhanes')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/nhanes_cohort.csv')
    args = parser.parse_args()
    # The production staging step read and rewrote the first prepared CSV.
    # Preserve that round trip to reproduce its floating-point serialization.
    prepared = build(args.data).to_csv(index=False)
    content = pd.read_csv(io.StringIO(prepared)).to_csv(index=False).encode()
    digest = hashlib.sha256(content).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f'Generated cohort differs from archived bytes: {digest}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    print(f'Verified 55,800 participants, 7,260 deaths, 23 groups: {args.output}')
    print('No fits or bootstraps were run.')


if __name__ == '__main__':
    main()
