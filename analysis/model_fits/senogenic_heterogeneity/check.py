"""Optional production/fine-grid check of an archived profile and anchor."""
import argparse
import json
import os
import time
from model import Model, BASE, HERE, objective
from fit import dataset, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    if not args.recompute:
        parser.error('Numerical reevaluation requires --recompute.')
    index = int(os.environ.get('LSB_JOBINDEX', '1'))-1
    record = json.loads((HERE/'profiles'/f'{index:02d}.json').read_text())
    d, e = dataset()
    results = []
    for name, params in [('refit', record['params']), ('anchored', BASE)]:
        for level in ['production', 'fine']:
            model = Model(record['tau_cv'], level)
            start = time.monotonic()
            rates = model.rates(params)
            results.append(dict(name=name, level=level, score=objective(model, params, d, e),
                                rates=rates.tolist(), seconds=time.monotonic()-start))
    save(HERE/'checks'/f'{index:02d}.json', results)


if __name__ == '__main__':
    main()
