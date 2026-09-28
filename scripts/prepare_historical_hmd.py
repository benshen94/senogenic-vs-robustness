#!/usr/bin/env python3
"""Build historical deaths/exposure inputs from bundled HMD period tables."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'SWE': (
        ('SWE_Deaths_1x1.txt', '1f91e4ca1ca4e2ba22f4c67ca950525f11c34f8bdd6a9b6a0c4915c6e5b4dcf9'),
        ('SWE_Exposures_1x1.txt', 'cf4e8adfb40c1ba005d5e549b6e895751a955905263e74352656a1529667697d'),
    ),
    'DNK': (
        ('DNK_Deaths_1x1_20260923.txt', 'bef68ddc699907681917ba1c7cd964a74887d40372a3bbb6a7d0454fb836fef6'),
        ('DNK_Exposures_1x1_20260923.txt', 'b5e498c18db495477d1b284a62dc60290ec5941f7b015b93ee131b450114ebb9'),
    ),
}


def build():
    countries = []
    for country, sources in SOURCES.items():
        tables = []
        for (name, checksum), value in zip(sources, ('deaths', 'exposure')):
            path = ROOT/'data/hmd'/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'Input checksum mismatch: {name}')
            frame = pd.read_csv(path, sep=r'\s+', skiprows=2)
            frame['open_ended'] = frame.Age.astype(str).str.endswith('+')
            frame['age'] = frame.Age.astype(str).str.rstrip('+').astype(int)
            frame = frame.rename(columns={'Year': 'year', 'Total': value})
            tables.append(frame[['year', 'age', 'open_ended', value]])
        frame = tables[0].merge(tables[1], on=['year', 'age', 'open_ended'], validate='one_to_one')
        frame = frame[frame.year.between(1800, 2019)].copy()
        frame.insert(0, 'country', country)
        countries.append(frame.sort_values(['year', 'age']))
    return pd.concat(countries, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/historical_inputs/hmd.csv')
    parser.add_argument('--compare', type=Path, help='Verify every value against an archived fitting input.')
    args = parser.parse_args()
    frame = build()
    if args.compare:
        pd.testing.assert_frame_equal(frame, pd.read_csv(args.compare), check_exact=True)
        print('All rows and values match the archived fitting input.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'Wrote {len(frame)} rows to {args.output}; no fits were run.')


if __name__ == '__main__':
    main()
