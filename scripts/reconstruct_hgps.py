#!/usr/bin/env python3
"""Verify bundled HGPS records against the published supplement; no fitting."""
import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = '7c4b2a50ca70b663856b29fed53de50b1ea994f850e553024fa1302b7ca2e733'


def extract(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Supplement differs from the archived source PDF.')
    text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
    records = []
    for label, heading, following, death in (
        ('censored', 'Table S1: Censored Subjects', 'Table S2:', 0),
        ('deceased', 'Table S3. Deceased Untreated Cohort', 'Table S4:', 1),
    ):
        start = text.rfind(heading)
        end = text.find(following, start)
        if start < 0 or end < 0:
            raise ValueError(f'Missing table boundaries: {heading}')
        for line in text[start:end].splitlines():
            pattern = r'HGPS(\d+)\s+([FM])\b(.*?)(?=\s+HGPS\d+\s+[FM]\b|$)'
            for match in re.finditer(pattern, line):
                nums = re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', match.group(3))
                if nums:
                    records.append(dict(id='HGPS'+match.group(1), sex=match.group(2),
                                        age=float(nums[-1]), death=death, table=label))
    if len(records) != 204 or len({r['id'] for r in records}) != 204:
        raise ValueError('Expected 204 unique source records.')
    if sum(r['death'] for r in records) != 102:
        raise ValueError('Expected 102 deaths.')
    return sorted(records, key=lambda r: (r['age'], r['id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('supplement', type=Path)
    parser.add_argument('--output', type=Path, help='Optional reconstructed CSV; must not exist.')
    args = parser.parse_args()
    records = extract(args.supplement)
    with (ROOT/'data/hgps/records.csv').open() as handle:
        bundled = list(csv.DictReader(handle))
    bundled = [dict(r, age=float(r['age']), death=int(r['death'])) for r in bundled]
    if records != bundled:
        raise ValueError('Reconstruction differs from bundled fitting records.')
    if args.output:
        with args.output.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'sex', 'age', 'death', 'table'])
            writer.writeheader()
            writer.writerows(records)
    print('Verified all 204 records (102 deaths, 102 censored) against the source PDF.')


if __name__ == '__main__':
    main()
