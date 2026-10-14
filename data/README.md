# Data Directory

This directory is designated for project datasets.

## Important Guidelines
- **Local Storage Only:** Clinical and research datasets must remain local and are excluded from Git version control via `.gitignore`.
- **Primary Expected Dataset:** `FraminghamDataset.csv` (or Framingham longitudinal study dataset).
- **Integrity:** Raw data files placed here must never be modified in-place; all transformations must be handled via reproducible pipelines in `src/` or documented in `notebooks/`.
