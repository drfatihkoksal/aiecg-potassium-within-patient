# AI-ECG potassium models and within-patient potassium change

Analysis code for a study evaluating whether artificial intelligence–enabled
electrocardiography (AI-ECG) models of serum potassium track potassium change
within the same patient, using serial 12-lead ECGs from MIMIC-IV.
Three models are evaluated: a 1-dimensional ResNet trained from scratch,
the ECGFounder foundation model fine-tuned on the same data, and a frozen
published potassium model trained in Sweden.

The repository contains code only. Patient-level data, model weights and
results are not redistributed.

## Data

| Source | Version | Access |
|---|---|---|
| [MIMIC-IV](https://doi.org/10.13026/kpb9-mt58) (hosp, icu) | 3.1 | Credentialed access via PhysioNet (PhysioNet Credentialed Health Data Use Agreement 1.5.0) |
| [MIMIC-IV-ECG](https://doi.org/10.13026/4nqg-sb35) | 1.0 | Open access via PhysioNet (Open Database License) |

Set the local paths to these datasets in `src/paths.py`.

## External models

| Model | Source | Location expected by the code |
|---|---|---|
| ECGFounder (12-lead) | [Hugging Face: PKUDigitalHealth/ECGFounder](https://huggingface.co/PKUDigitalHealth/ECGFounder) | `checkpoint/12_lead_ECGFounder.pth` |
| Stockholm potassium regression models | [Zenodo 10.5281/zenodo.7456316](https://doi.org/10.5281/zenodo.7456316) | `checkpoint/stockholm/` |
| Stockholm model code | [philippvb/ecg-electrolyte-regression](https://github.com/philippvb/ecg-electrolyte-regression) @ `be88849` | `checkpoint/ecg-electrolyte-regression/` |
| Stockholm preprocessing | [antonior92/ecg-preprocessing](https://github.com/antonior92/ecg-preprocessing) @ `439bc1b` | `checkpoint/ecg-preprocessing/` |

## Pipeline

Run the scripts in `src/` in this order (each stage reads the previous stage's
outputs from `data/interim/` and writes tables to `outputs/`):

| Stage | Scripts | Purpose |
|---|---|---|
| Feasibility | `wp0_00` – `wp0_04` | Laboratory extraction, dialysis sessions, protocol feasibility gate |
| Cohorts | `wp1_01` – `wp1_03` | ECG–potassium observations, serial pairs, patient-disjoint development pool |
| In-house model | `wp2_01` – `wp2_05`, `wp3_01` | Waveform extraction and filtering, training, baselines, prediction |
| Core analyses | `wp3_02`, `wp4_dt_strat`, `wp6_01`, `wp6_02`, `wp7_01` | Change-scale agreement, interval strata, specificity, matched cross-patient comparison |
| ECGFounder | `ecgf_00` – `ecgf_03` | Checkpoint preparation, preprocessing, fine-tuning, prediction |
| Stockholm model | `stockholm_01` – `stockholm_03` | Frozen inference with the authors' preprocessing, comparison, report |
| Exploratory | `wp8_*`, `delta_01_experiment` | Change-specific models (not part of the main analysis) |
| Manuscript analyses | `wp9_01` – `wp9_03` | Direction vs magnitude analysis, figures, reference list |

Shared code: `paths.py` (locations), `wp_stats.py` (patient-level bootstrap,
Deming regression, intraclass correlation), `wp2_model.py` (ResNet),
`ecgf_model.py` and `net1d.py` (ECGFounder architecture).

## Environment

Python 3.13 with the packages in `requirements.txt` (PyTorch with CUDA was used
for training).

## Third-party code

`src/net1d.py` is from the [ECGFounder repository](https://github.com/PKUDigitalHealth/ECGFounder)
(MIT License); see `THIRD_PARTY_NOTICES.md`.

## Citation

Archived on Zenodo: [doi:10.5281/zenodo.23737739](https://doi.org/10.5281/zenodo.23737739) (all versions); v1.0.0: [doi:10.5281/zenodo.23737740](https://doi.org/10.5281/zenodo.23737740). See `CITATION.cff`.

## License

MIT License (see `LICENSE`). Third-party code keeps its own license (see `THIRD_PARTY_NOTICES.md`).
