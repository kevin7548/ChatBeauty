"""
Zip what colab_dense.ipynb needs into experiments/B/cache/chatbeauty_B_bundle.zip
(repo-relative layout, so common.py paths work unchanged on Colab).

Usage:
    python experiments/B/make_colab_bundle.py
"""

import zipfile

from common import B_DIR, CACHE, ITEMS_PATH, ML, QUERIES, REPO, SPLIT

FILES = [
    B_DIR / 'common.py',
    B_DIR / 'dense_retrieve.py',
    ML / 'evaluation/retrieve_candidates.py',
    ITEMS_PATH,
    QUERIES,
]


def main():
    out = CACHE / f"chatbeauty_B_bundle_{SPLIT}.zip"
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for path in FILES:
            z.write(path, path.relative_to(REPO))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == '__main__':
    main()
