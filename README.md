# descriptor-limitations

Quantifying fundamental limits of ML prediction in chemistry and materials science via information-theoretic accuracy ceilings (R²_ceiling, Fano).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
```

## Structure

- `src/descriptor_limitations/` — core library (`information.py`, `data_loaders.py`, `plotting.py`)
- `notebooks/` — marimo notebooks, one per case study
- `tests/` — pytest suite
- `data/` — raw datasets (gitignored; loader scripts commit)
- `figures/` — publication figures (regenerated from scripts)
- `paper/` — LaTeX source

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.
`marimo` for notebooks — pure-Python, reactive, diffable; chosen over Jupyter for reproducibility and reviewability.
