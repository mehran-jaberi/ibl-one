# ibl-one

Work with **International Brain Laboratory (IBL)** neurophysiology data — download sessions via the ONE API, separate visual vs. somatosensory neurons, run exploratory analyses, and decode visual stimuli from population activity with scikit-learn.

## Overview

This project is a step-by-step pipeline for working with the IBL public dataset, using the [ONE API](https://github.com/int-brain-lab/ONE) to access the public data server at `https://openalyx.internationalbrainlab.org`.

The main script, `IBL_Main.py`, is organized into 13 numbered sections that build on each other:

| Section | Description |
| --- | --- |
| **1. Setup** | Connect to the IBL public data server (ONE API) |
| **2. Search** | Find sessions by date range, subject, lab, project, tags, or datasets |
| **3. Session details** | Get metadata for a session (`subject`, `start_time`, `lab`, `task_protocol`, …) |
| **4. List datasets** | See available ALF datasets for a session (incl. per-probe) |
| **5. Load data** | Load spikes, trials, wheel, clusters, camera (DeepLabCut), and raw ephys data |
| **6. Download only** | Download datasets to the local cache without loading into memory |
| **7. Quality control** | Load only QC-passed datasets (`qc='WARNING'`) |
| **8. Cache management** | Refresh the ONE cache tables with `one.load_cache()` |
| **9. Work offline** | Use a local cache directory without network access |
| **10. Batch loading** | Load trials across many sessions in a loop |
| **11. Vision vs. somatosensory split** | Classify clusters by Allen brain region and separate spike trains by modality |
| **12. Analysis & visualization** | Firing-rate stats, PSTHs, rasters, contrast tuning, region composition |
| **13. ML decoding** | Predict stimulus presence, side, and contrast from neural activity |

## Features

- **Data access** — search, download, and load IBL sessions (ephys, behavior, wheel, camera) via the ONE API.
- **Modality separation** — classifies each recorded cluster into *vision* (visual cortex, visual thalamus, superior colliculus), *somatosensory* (barrel cortex, somatosensory thalamus, trigeminal nuclei), or *other* using Allen CCF region acronyms.
- **Exploratory analysis** — firing-rate distributions, stimulus-aligned PSTHs, spike rasters, contrast-response curves, and cross-session region composition (saved as `fig12*.png`).
- **Machine-learning decoding** — trial-by-trial spike-count features compared across three feature sets (vision-only, somatosensory-only, combined) and five classifiers:
  - Logistic Regression (L2)
  - Linear SVM
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (2 hidden layers)
- **Three decoding tasks**:
  - **Task A** — Stimulus detection (binary: stimulus present?)
  - **Task B** — Stimulus side (left vs. right, stimulus trials only)
  - **Task C** — Contrast prediction (regression)
- **Rigorous evaluation** — 5-fold stratified cross-validation with accuracy, F1, ROC-AUC (classification), R²/MAE (regression), and **permutation tests** for significance against chance.

## Installation

Requires Python **≥ 3.10**.

```bash
# Install the project with dependencies
pip install -e .

# Or install the core API + analysis stack manually
pip install ONE-api pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

Dependencies (from `pyproject.toml`): `lazypredict`, `matplotlib`, `mne`, `one-api`, `seaborn` (+ `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels` used in the script).

## Usage

Run the full pipeline:

```bash
python IBL_Main.py
```

> **Note**: the script downloads real IBL data on first run, so the initial execution takes time depending on how many sessions are processed. Sections 11–13 default to a small subset (e.g. `eids_ephys[:5]` and the first session) for demo purposes — remove the slicing to analyze all sessions.

### Outputs

Figures are saved to the working directory:

| File | Content |
| --- | --- |
| `fig12c_firing_rates.png` | Firing-rate distributions, vision vs. somatosensory |
| `fig12d_psth.png` | PSTH aligned to visual-stimulus onset |
| `fig12e_raster.png` | Spike rasters around stimulus onset |
| `fig12f_contrast_tuning.png` | Contrast-response function (vision neurons) |
| `fig12g_region_composition.png` | Cluster composition by brain region |
| `fig13g_decoding_summary.png` | Best decoding performance per task × feature set |
| `fig13h_model_heatmap.png` | Model × feature-set detection accuracy |
| `fig13i_confusion_matrix.png` | Confusion matrix for the final detection model |
| `fig13j_neuron_weights.png` | Per-neuron decoder weights (vision vs. somato) |

## Project structure

```
├── IBL_Main.py       # Main analysis pipeline (sections 1–13)
├── main.py           # Minimal placeholder entry point
├── pyproject.toml    # Project metadata & dependencies
└── README.md
```

## Data & acknowledgments

- Public IBL data server: `https://openalyx.internationalbrainlab.org` (public password: `international`)
- Data format: [ALF (Alyx Little Format)](https://docs.internationalbrainlab.org/)
- Brain regions use Allen Institute Common Coordinate Framework (CCF) acronyms.

Please cite the relevant IBL papers and the ONE API when using this data in publications.