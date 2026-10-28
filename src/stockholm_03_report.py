"""Render the Stockholm comparison from saved, verified numerical outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'outputs'

def main():
    data=json.loads((O/'stockholm_comparison.json').read_text());records={r['model']:r for r in data['models']}
    names=['potassium','ecgfounder','stockholm_3'];labels=['In-house ResNet','ECGFounder fine-tuned','Published Stockholm model 3']
    r=records['stockholm_3'];c=data['calibration']['stockholm_3']
    lines=['# Published potassium checkpoint: external evaluation on MIMIC-IV', '',
    'The frozen Swedish potassium model reproduces weak serial tracking and predominantly between-patient explained variance. This is external evaluation of a Swedish-trained model on MIMIC; the outcome dataset remains single-centre.', '',
    '## Source and predeclared comparison', '',
    '[von Bachmann et al., Scientific Reports 2024](https://doi.org/10.1038/s41598-024-65223-w); [released weights](https://doi.org/10.5281/zenodo.7456316); [author code](https://github.com/philippvb/ecg-electrolyte-regression). The Zenodo description identifies regression model 3 as the model used for the potassium graphics. It was selected before observing MIMIC performance. All five released regression checkpoints were evaluated as sensitivity analyses; none was selected using MIMIC test performance.', '',
    'The checkpoint has 6,920,353 parameters, including its published potassium head. All weights and batch-normalisation state stayed frozen. Input: I, II, V1–V6 in mV; native-rate elliptic high-pass filtering (0.8 Hz cutoff, 0.2 Hz stopband, 0.5 dB ripple, 40 dB attenuation); 50 Hz notch, Q=30; polyphase resampling from 500 to 400 Hz; 48 zero samples at each end, yielding 8 × 4096. The 50 Hz notch follows the published protocol; MIMIC was not used to optimise filtering. This differs from the US mains frequency and remains a transport limitation. Numerical checks against the cited author preprocessing were exactly equal after float32 conversion on three selected records.', '',
    '## Denominators and primary results without local calibration', '',
    '18,563 test ECGs from 9,490 patients; 15,279 ECGs with prior laboratory history; 17,571 unique audit ECGs from 7,453 patients; 9,154 serial pairs; 1,817 near-stable pairs from 1,707 patients. These are the fixed existing cohorts. A total of 182,713 unique ECGs were scored to cover the test, audit, and calibration-training sets.', '',
    '| Metric | In-house ResNet | ECGFounder | Published Stockholm model 3 |', '|---|---|---|---|']
    def row(label,key,fmt='.3f'):
        lines.append('| '+label+' | '+' | '.join(format(records[n][key],fmt) for n in names)+' |')
    row('Cross-sectional r','cross_sectional_r');row('AUROC K ≥ 5.5','auroc_K_ge_5_5')
    lines.append('| Within-patient Δ r (95% patient-bootstrap CI) | '+' | '.join(f"{records[n]['delta_r']:.3f} ({records[n]['delta_r_ci_low']:.3f}–{records[n]['delta_r_ci_high']:.3f})" for n in names)+' |')
    row('r between','r_between');row('r within','r_within')
    lines.append('| Within share of explained variance | '+' | '.join(f"{records[n]['within_explained_share']*100:.1f}%" for n in names)+' |')
    row('Prediction ICC','icc_prediction');row('Distribution-matched cross-patient r','matched_cross_r')
    lines+=['', 'All results in this table use untouched checkpoint scores. Pearson correlations, AUROC, ICC, and the explained-variance share are invariant to a positive affine conversion back to potassium units.', '',
    'The patient bootstrap uses 2,000 draws with the same sampled patients for every model. Stockholm model 3 minus ResNet: Δr difference −0.0256 (95% CI −0.0452 to −0.0067); minus ECGFounder: −0.0166 (−0.0380 to 0.0027). The published checkpoint has slightly lower Δ correlation than ResNet; its difference from ECGFounder is not resolved by this interval.', '',
    f"Distribution-matched cross-patient r is {r['matched_cross_r']:.3f}, versus within-patient r {r['delta_r']:.3f}. The matched procedure repeats the existing 20-bin histogram-matching algorithm on shared synthetic pairs. Its 200-repetition range ({r['matched_cross_mc_low']:.3f}–{r['matched_cross_mc_high']:.3f}) is a Monte Carlo percentile interval, **not** a patient-bootstrap confidence interval. The two interval types must not be interpreted as a paired significance test.", '',
    '## Missing source target normalisation and separate local calibration', '',
    'Every released regression configuration has normalize=true. The author loader standardises potassium using its Swedish training mean and SD, and the evaluation script normally reads those statistics from the private training CSV. The downloaded regression members contain a checkpoint and training configuration but no target mean/SD. Consequently, uncalibrated MAE, Deming slope in potassium units, and MDC in mEq/L cannot be recovered exactly from this release. No Swedish mean or SD was guessed.', '',
    f"For a separately labelled transport-with-recalibration analysis, ordinary least squares was fitted on the existing MIMIC training split only (146,579 ECGs; 75,886 patients): Khat = {c['intercept_mEq']:.9f} + {c['slope_mEq_per_z']:.9f} × z. No validation, test, or audit labels entered this fit. The backbone and potassium head were not retrained. The calibration coefficients were then fixed for testing. This changes the output mapping and is **not** pure uncalibrated external validation. Intervals for recalibrated MAE gains condition on the fitted coefficients.", '',
    '| Metric in potassium units | In-house native output | ECGFounder native output | Stockholm + MIMIC-training calibration |','|---|---|---|---|']
    row('Cross-sectional MAE (mEq/L)','mae_mEq');row('MAE on history-available test ECGs','history_subset_mae_mEq');row('Δ MAE (mEq/L)','delta_mae_mEq')
    lines.append('| Improvement over no change (MAE 0.643) | '+' | '.join(f"{records[n]['no_change_gain_percent']:.1f}%" for n in names)+' |')
    row('Deming slope','deming_slope_mEq');row('MDC95 (mEq/L)','mdc95_mEq')
    lines+=['', 'Historical-mean MAE is 0.388 on the same 15,279 test ECGs; full-test constant MAE is 0.498. The potassium near-stable definition is strict absolute ΔK < 0.15 mEq/L without recorded dialysis or CRRT. Residual physiological change can inflate MDC; its interpretation remains conditional on that definition.', '',
    '## All released seeds and supporting analyses', '',
    '| Released model | Δ r | Within share | Calibrated Δ MAE | Calibrated MDC95 |', '|---|---|---|---|---|']
    for i in range(5):
        q=records[f'stockholm_{i}'];lines.append(f"| {i}{' (primary)' if i==3 else ''} | {q['delta_r']:.3f} | {q['within_explained_share']*100:.1f}% | {q['delta_mae_mEq']:.3f} | {q['mdc95_mEq']:.3f} |")
    lines+=['', '**The weak tracking and between-patient predominance recur across all five checkpoints. The clinical MDC threshold claim does not:** model 4 gives calibrated MDC95 0.463, below 0.5, despite weak tracking. A compressed-output model can have low MDC without adequate responsiveness. MDC must be interpreted jointly with tracking and calibration.', '',
    'For model 3, Δ correlations across <2, 2–6, 6–12, and 12–24 hours are 0.108, 0.303, 0.340, and 0.363. The final interval includes exactly 24 hours, so all 9,154 pairs are retained (the earlier Table 7 omitted three boundary pairs). This remains descriptive, not a causal test of elapsed time.', '',
    'The specificity sensitivity uses original analyte labels and exact ECG-pair intersections: ΔK r 0.331 (n=9,154), Δcreatinine 0.193 (n=8,472), ΔHb 0.057 (n=5,787), ΔCa −0.009 (n=4,075). These are different denominators and a different label-extraction protocol from the original Table 6; the latter re-extracts nearest laboratory values. They are not interchangeable.', '',
    '## Reproducibility', '',
    'Run `python src/stockholm_01_predict.py`, then `python src/stockholm_02_compare.py`, then `python src/stockholm_03_report.py` in the existing project environment. Raw waveforms and existing cohort files are prerequisites. Downloaded author repositories are pinned by commit in `pred_stockholm_provenance.json`; checkpoint CRC verification and SHA-256 hashes are recorded in `checkpoint/stockholm/manifest.json`. Only potassium regression members were extracted from the Zenodo archive using HTTP byte ranges; the complete archive MD5 was not verified.', '',
    'Outputs: [comparison table](table12_stockholm_comparison.csv), [paired differences](table12d_stockholm_paired_differences.csv), [Δt strata](table12b_stockholm_dt.csv), [specificity](table12c_stockholm_specificity.csv), [complete numeric record](stockholm_comparison.json), [inference provenance](pred_stockholm_provenance.json).', '',
    '![Frozen published checkpoint comparison](fig5_stockholm_comparison.png)', '']
    (O/'stockholm_comparison.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),gridspec_kw={'width_ratios':[1.5,1]})
    cols=['#35618D','#159895','#C76835'];yy=np.arange(3)
    for j,name in enumerate(names):
        q=records[name];x=q['delta_r'];axes[0].errorbar(x,j,xerr=[[x-q['delta_r_ci_low']],[q['delta_r_ci_high']-x]],fmt='o',color=cols[j],capsize=4,markersize=7)
        axes[0].text(.402,j,f"{x:.3f}",va='center',color=cols[j])
    axes[0].set(yticks=yy,yticklabels=labels,xlim=(.28,.43),xlabel='Within-patient delta K correlation',title='9,154 pairs · patient-bootstrap 95% CI');axes[0].invert_yaxis()
    shares=[records[n]['within_explained_share']*100 for n in names]
    axes[1].barh(yy,shares,color=cols,height=.55);axes[1].set(yticks=yy,yticklabels=[],xlim=(0,100),xlabel='Within-patient share (%)',title='Share of explained variance');axes[1].invert_yaxis()
    for j,x in enumerate(shares):axes[1].text(x+2,j,f'{x:.1f}%',va='center')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Swedish-trained potassium checkpoint evaluated on MIMIC',fontsize=13)
    fig.tight_layout();fig.savefig(O/'fig5_stockholm_comparison.svg');fig.savefig(O/'fig5_stockholm_comparison.png',dpi=180);plt.close(fig)
    print('Wrote stockholm_comparison.md and Figure 5.')

if __name__=='__main__':main()
