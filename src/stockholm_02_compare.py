"""Comparison of frozen Swedish outputs and explicitly separate MIMIC calibration.
No test/audit labels enter calibration. Patient-cluster bootstrap is paired
across models. Matched-shuffle percentiles describe shuffle Monte Carlo variation,
not patient-bootstrap confidence intervals.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import duckdb
from wp_stats import icc_oneway, deming, bland_altman

ROOT=Path(__file__).resolve().parents[1];I=ROOT/'data/interim';O=ROOT/'outputs'
NBOOT=2000

def correlation(x,y):return float(np.corrcoef(x,y)[0,1])
def corr_cols(x,y):
    x=x-x.mean();y=y-y.mean(axis=0)
    return (x[:,None]*y).sum(axis=0)/np.sqrt((x*x).sum()*(y*y).sum(axis=0))
def interval(v):return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]

def main():
    raw=pd.read_parquet(I/'pred_stockholm.parquet')
    sp=pd.read_parquet(I/'split_potassium.parquet')
    d=pd.read_parquet(I/'audit_potassium.parquet')
    nl=pd.read_parquet(I/'null_potassium.parquet')
    pred=raw[['study_id']].copy()
    names=['potassium','ecgfounder']+[f'stockholm_{i}' for i in range(5)]
    for name in names[:2]:
        pr=pd.read_parquet(I/f'pred_{name}.parquet')[['study_id',f'pred_{name}']].rename(columns={f'pred_{name}':name})
        pred=pred.merge(pr,on='study_id',validate='one_to_one')
    calibration={}
    for seed in range(5):
        name=f'stockholm_{seed}';pred[name]=raw[f'{name}_z'].values.astype(float)
        tr=sp[sp.split=='train'].merge(pred[['study_id',name]],on='study_id',validate='one_to_one')
        assert len(tr)==146579
        assert not set(tr.subject_id)&set(d.subject_id)
        assert not set(tr.subject_id)&set(sp.loc[sp.split=='test','subject_id'])
        slope,intercept=np.polyfit(tr[name],tr.y,1)
        assert slope>0
        calibration[name]={'slope_mEq_per_z':float(slope),'intercept_mEq':float(intercept),'n_training_ecgs':len(tr),'n_training_patients':int(tr.subject_id.nunique()),'fit':'ordinary least squares on MIMIC training split only; frozen backbone and head'}
    te=sp[sp.split=='test'].merge(pred,on='study_id',validate='one_to_one');assert len(te)==18563
    con=duckdb.connect(str(I/'wp0.duckdb'),read_only=True);con.execute('pragma threads=8')
    con.register('te',te[['subject_id','study_id','ecg_time']])
    pm=con.execute("SELECT te.study_id,(SELECT avg(valuenum) FROM labs l WHERE l.subject_id=te.subject_id AND l.series='k_chem' AND l.charttime<te.ecg_time) patient_mean FROM te").df()
    te=te.merge(pm,on='study_id',validate='one_to_one');hist=te.patient_mean.notna();assert hist.sum()==15279
    for name in names:
        lookup=pred.set_index('study_id')[name]
        d[name+'_pre']=d.pre_study.map(lookup);d[name+'_post']=d.post_study.map(lookup)
        d[name]=d[name+'_post']-d[name+'_pre']
        nl[name]=nl.post_study.map(lookup)-nl.pre_study.map(lookup)
    assert len(d)==9154 and d[names].notna().all().all() and len(nl)==1817
    obs=pd.concat([d[['subject_id','pre_study','y_pre']].rename(columns={'pre_study':'study_id','y_pre':'y'}),d[['subject_id','post_study','y_post']].rename(columns={'post_study':'study_id','y_post':'y'})]).drop_duplicates('study_id').merge(pred,on='study_id',validate='one_to_one')
    assert len(obs)==17571 and obs.subject_id.nunique()==7453
    y=d.dy_lab.to_numpy(float); Y=d[names].to_numpy(float)
    Yunits=Y.copy()
    for j,name in enumerate(names):
        if name in calibration:Yunits[:,j]*=calibration[name]['slope_mEq_per_z']
    # Paired cluster bootstrap using sufficient statistics (same sampled patients).
    _,inv=np.unique(d.subject_id,return_inverse=True);g=inv.max()+1
    arr=np.column_stack([np.ones(len(d)),y,y*y,Y,Y*Y,y[:,None]*Y,np.abs(Yunits-y[:,None]),np.abs(y)])
    sums=np.zeros((g,arr.shape[1]));np.add.at(sums,inv,arr)
    m=len(names);rng=np.random.default_rng(11);br=[];bmae=[];bnc=[]
    for _ in range(NBOOT):
        counts=np.bincount(rng.integers(0,g,g),minlength=g);v=counts@sums
        n,sx,sxx=v[:3];sy=v[3:3+m];syy=v[3+m:3+2*m];sxy=v[3+2*m:3+3*m]
        br.append((sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n)))
        bmae.append(v[3+3*m:3+4*m]/n);bnc.append(v[-1]/n)
    br=np.array(br);bmae=np.array(bmae);bnc=np.array(bnc)
    np.testing.assert_allclose(corr_cols(y,Y),[correlation(y,Y[:,j]) for j in range(m)],atol=1e-12)
    # Match the existing 20-quantile-bin shuffle on identical synthetic pairs.
    edges=np.quantile(y,np.linspace(0,1,21));edges[0],edges[-1]=-np.inf,np.inf
    assert np.all(np.diff(edges)>0)
    target=np.histogram(y,bins=edges)[0];rng=np.random.default_rng(7);matched=[];unmatched=[]
    sid=d.subject_id.to_numpy();pre=d[[n+'_pre' for n in names]].to_numpy();post=d[[n+'_post' for n in names]].to_numpy()
    for _ in range(200):
        ii=rng.integers(0,len(d),len(d)*6);jj=rng.integers(0,len(d),len(d)*6);ok=sid[ii]!=sid[jj];ii,jj=ii[ok],jj[ok]
        x=d.y_post.to_numpy()[jj]-d.y_pre.to_numpy()[ii];ys=post[jj]-pre[ii]
        unmatched.append(corr_cols(x,ys));bins=np.digitize(x,edges)-1;sel=[]
        for k,want in enumerate(target):
            cand=np.flatnonzero(bins==k)
            assert len(cand) or not want
            if want:sel.append(rng.choice(cand,size=want,replace=len(cand)<want))
        sel=np.concatenate(sel);assert len(sel)==len(d)
        matched.append(corr_cols(x[sel],ys[sel]))
    matched=np.array(matched);unmatched=np.array(unmatched)
    summaries=[];dtrows=[];specificity=[]
    constant=float(sp.loc[sp.split=='train','y'].mean());nc=float(np.abs(y).mean())
    for j,name in enumerate(names):
        c=calibration.get(name,{'slope_mEq_per_z':1.,'intercept_mEq':0.})
        zz=te[name].to_numpy();ph=c['intercept_mEq']+c['slope_mEq_per_z']*zz
        rep=obs[['subject_id','y',name]].copy();grp=rep.groupby('subject_id');yb=grp.y.transform('mean');pb=grp[name].transform('mean');pm=grp[['y',name]].mean()
        rb=correlation(pm.y,pm[name]);rw=correlation(rep.y-yb,rep[name]-pb)
        eb=rb**2*pm.y.var(ddof=1);ew=rw**2*(rep.y-yb).var(ddof=1)
        sd=float(nl[name].std(ddof=1)/np.sqrt(2));mdc=1.96*np.sqrt(2)*sd
        r=correlation(y,Y[:,j]);lo,hi=interval(br[:,j]);rlo,rhi=interval(matched[:,j])
        dy=Yunits[:,j];mae=float(np.abs(dy-y).mean());calibrated=name in calibration
        record={'model':name,'primary':name=='stockholm_3','scale_status':'MIMIC-train-only affine calibration for mEq metrics' if calibrated else 'native mEq/L model', 'n_test':len(te),'n_history':int(hist.sum()),'n_pairs':len(d),'n_patients':int(d.subject_id.nunique()),'n_obs':len(obs),'n_null':len(nl), 'cross_sectional_r':correlation(te.y,zz),'auroc_K_ge_5_5':float(roc_auc_score(te.y>=5.5,zz)),'mae_mEq':float(np.abs(ph-te.y).mean()),'history_subset_mae_mEq':float(np.abs(ph[hist]-te.y[hist]).mean()),'r_between':rb,'r_within':rw,'within_explained_share':float(ew/(eb+ew)),'icc_prediction':icc_oneway(rep[name],rep.subject_id)['icc'],'delta_r':r,'delta_r_ci_low':lo,'delta_r_ci_high':hi,'delta_mae_mEq':mae,'no_change_mae_mEq':nc,'no_change_gain_percent':100*(1-mae/nc),'no_change_gain_percent_ci':interval(100*(1-bmae[:,j]/bnc)),'deming_slope_mEq':deming(y,dy)[0],'null_sd_output_units':sd,'mdc95_output_units':mdc,'mdc95_mEq':mdc*c['slope_mEq_per_z'],'matched_cross_r':float(matched[:,j].mean()),'matched_cross_mc_low':rlo,'matched_cross_mc_high':rhi,'matched_interval_kind':'Monte Carlo shuffle percentiles, not patient-bootstrap CI','raw_cross_r':float(unmatched[:,j].mean()),'bland_altman_mEq':bland_altman(dy,y)}
        summaries.append(record)
        for a,b,label in [(0,2,'<2h'),(2,6,'2-6h'),(6,12,'6-12h'),(12,24,'12-24h inclusive')]:
            q=(d.dt_h>=a)&((d.dt_h<=b) if b==24 else (d.dt_h<b))
            dtrows.append({'model':name,'stratum':label,'n_pairs':int(q.sum()),'r':correlation(y[q],Y[q,j]),'mean_abs_delta_K':float(np.abs(y[q]).mean())})
        # Use existing primary analyte cohorts, exact pair intersection; no re-extraction.
        for analyte in ['potassium','creatinine','hemoglobin','calcium_total']:
            other=pd.read_parquet(I/f'delta_scored_{analyte}.parquet')[['pre_study','post_study','dy_lab']]
            joint=d[['pre_study','post_study',name]].merge(other,on=['pre_study','post_study'],validate='one_to_one')
            specificity.append({'model':name,'target':analyte,'n_pairs':len(joint),'r':correlation(joint[name],joint.dy_lab),'labels':'original analyte audit labels; exact ECG-pair intersection'})
    comparisons=[];j=names.index('stockholm_3')
    for other in names[:2]:
        k=names.index(other);point=summaries[j]['delta_r']-summaries[k]['delta_r'];lo,hi=interval(br[:,j]-br[:,k])
        comparisons.append({'comparison':'stockholm_3 minus '+other,'paired_delta_r_difference':point,'ci_low':lo,'ci_high':hi,'n_patient_bootstraps':NBOOT})
    result={'primary':'stockholm_3','source_target_mean_sd_available':False,'physical_units_warning':'All Stockholm mEq/L metrics use a separate OLS calibration fitted only on MIMIC training data; they are not uncalibrated external-validation performance. Correlations/AUROC/decomposition use untouched standardized checkpoint outputs.','bootstrap':'2000 paired patient-cluster resamples, seed11','calibration':calibration,'models':summaries,'paired_comparisons':comparisons,'baselines':{'constant_training_mean':constant,'constant_test_mae':float(np.abs(te.y-constant).mean()),'history_mean_mae':float(np.abs(te.loc[hist,'patient_mean']-te.loc[hist,'y']).mean()),'no_change_mae':nc},'dt_bins_note':'Final bin includes exactly24h; all9154pairs retained.'}
    (O/'stockholm_comparison.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(summaries).to_csv(O/'table12_stockholm_comparison.csv',index=False)
    pd.DataFrame(dtrows).to_csv(O/'table12b_stockholm_dt.csv',index=False)
    pd.DataFrame(specificity).to_csv(O/'table12c_stockholm_specificity.csv',index=False)
    pd.DataFrame(comparisons).to_csv(O/'table12d_stockholm_paired_differences.csv',index=False)
    d.to_parquet(I/'delta_scored_stockholm_z.parquet',index=False)
    print(pd.DataFrame(summaries)[['model','cross_sectional_r','auroc_K_ge_5_5','delta_r','within_explained_share','delta_mae_mEq','mdc95_mEq']].to_string(index=False),flush=True)
    print(json.dumps(comparisons,indent=2),flush=True)

if __name__=='__main__':main()
