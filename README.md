# MMMU‑Pro Visualizer (Vision ↔ Standard 10)

Side‑by‑side HTML visualizer for MMMU/MMMU_Pro that pairs the Vision split with the Standard (10 options) split by `id`. It lists the actual multiple‑choice options, anchors in‑question `<image k>` placeholders to figures, and supports MathJax, incremental rendering, and multi‑process image exporting.

## Features
- Pairs Vision vs. Standard(10) by `id` with consistent metadata.
- Correctly renders the 10 options on the right panel.
  - Parses `options` even when it is a stringified list.
  - If `options` only contains letters A–J, it reconstructs texts from the question when present; otherwise shows the original letter order.
  - Anchors `<image k>` options to the first matching figure.
- MathJax for inline formulas and lazy, incremental page rendering.
- Multi‑process export of original images (PNG/JPEG) with sensible defaults.
- Client‑side filters: subject, image type, difficulty, plus full‑text search.

## Install
- Python 3.9+
- `pip install -r requirements.txt`

## Usage
Run from the repo root:

```
python visualize_mmmu_pro_pairs_idjoin_mathjax_mp_optfix.py \
  --rows 200 --page-size 60 --workers 8
```

Common flags:
- `--subject SUBJECT` filter a specific subject.
- `--start N` start index; `--rows N` max items to render; `--page-size N` items per virtual page.
- `--workers N` processes for saving images; `--jpeg-quality`/`--jpeg-optimize`/`--jpeg-progressive` control output.
- `--out DIR` output folder, default `mmmu_pro_pairs_idjoin_mathjax_mp_options_listed` (contains `index.html` + `images/`).

Open the generated `index.html` in a browser. Use Ctrl+F5 to avoid stale caches when iterating.

## Notes
- The script auto‑detects image columns (`image`, `image_1..image_10`), saves originals to `images/`, and links them.
- It tolerates different column suffixes after merging (`*_v`/`*_s`), always preferring the Standard side for question/options/meta.
- If an `<image k>` placeholder has no matching `image_k` column, it is flagged in the yellow warning bar.

## Data & Credits
This tool downloads data from the Hugging Face dataset:
- `MMMU/MMMU_Pro` (Vision split and Standard (10 options) split).

Please follow the dataset’s license/terms when using the data. Cite the MMMU/MMMU_Pro paper/dataset where appropriate.

## License
MIT — see `LICENSE`.

