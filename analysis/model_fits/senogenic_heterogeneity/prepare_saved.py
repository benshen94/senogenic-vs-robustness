"""Copy archived profile candidates to the optional rerun directory; no fits."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'results/senogenic_heterogeneity'
DEST = ROOT/'tmp/senogenic_refit'


if __name__ == '__main__':
    if DEST.exists():
        raise SystemExit(f'Rerun directory already exists: {DEST}; no files copied.')
    DEST.mkdir(parents=True)
    for name in ('profiles', 'selected', 'stress', 'checks'):
        shutil.copytree(SOURCE/name, DEST/name)
    for name in ('zero_production_refit.json', 'four_wide_refit.json'):
        shutil.copy2(SOURCE/name, DEST/name)
    print(f'Staged saved candidates at {DEST}; no calculation started.')
