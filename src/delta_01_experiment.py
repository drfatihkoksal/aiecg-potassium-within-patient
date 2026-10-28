"""Exploratory direct-delta learning; see outputs/delta_learning/protocol.md."""
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='8'
import json, copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
import joblib
from paths import INTERIM

O=Path(__file__).resolve().parents[1]/'outputs/delta_learning'
O.mkdir(exist_ok=True)
torch.set_num_threads(8)
ALPHAS=[1,10,100,1000,10000]

def mae(y,p):return float(np.abs(y-p).mean())
def corr(y,p):return float(np.corrcoef(y,p)[0,1]) if np.std(p)>0 else None

def main():
 d=pd.read_parquet(INTERIM/'audit_potassium.parquet')
 original=len(d)
 d=d[(d.dt_h>0)&(d.dt_h<=24)&(d.t_pre<d.t_post)&(d.t_pre<=d.post_ecg)].copy().reset_index(drop=True)
 ids=np.sort(d.subject_id.unique());rng=np.random.default_rng(20260908);rng.shuffle(ids)
 n=len(ids);mapping={p:s for s,ps in [('train',ids[:int(.6*n)]),('val',ids[int(.6*n):int(.8*n)]),('test',ids[int(.8*n):])] for p in ps}
 d['split']=d.subject_id.map(mapping)
 original_train=pd.read_parquet(INTERIM/'split_potassium.parquet')
 assert not set(ids)&set(original_train.subject_id)
 for a,b in [('train','val'),('train','test'),('val','test')]:assert not set(d.loc[d.split==a,'subject_id'])&set(d.loc[d.split==b,'subject_id'])
 d.to_parquet(O/'pairs.parquet',index=False)
 counts=d.groupby('split').agg(pairs=('subject_id','size'),patients=('subject_id','nunique')).to_dict('index')
 print('COHORT',counts,'excluded',original-len(d),flush=True)
 pred=pd.read_parquet(INTERIM/'pred_potassium.parquet').set_index('study_id')
 e=np.load(INTERIM/'emb_potassium.npy',mmap_mode='r')
 pre=pred.loc[d.pre_study];post=pred.loc[d.post_study]
 e0=e[pre.row.to_numpy()].astype('float32');e1=e[post.row.to_numpy()].astype('float32')
 p0=pre.pred_potassium.to_numpy();p1=post.pred_potassium.to_numpy()
 y0=d.y_pre.to_numpy();y1=d.y_post.to_numpy();dy=y1-y0
 dt=np.log1p(d.dt_h.to_numpy())[:,None]
 x=np.column_stack([e1-e0,(e1+e0)/2,dt]).astype('float32')
 masks={s:(d.split==s).to_numpy() for s in ['train','val','test']};tr,va,te=[masks[s] for s in ['train','val','test']]
 history=[];saved={}
 def ridge_select(name,X,Y):
  best=None;score=np.inf
  for alpha in ALPHAS:
   model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
   model.fit(X[tr],Y[tr]);v=mae(Y[va],model.predict(X[va]));history.append(dict(model=name,alpha=alpha,val_mae=v))
   if v<score:score=v;best=model
  saved[name]=best;print(name,'validation MAE',score,flush=True)
  return best.predict(X),score
 with threadpool_limits(limits=8):
  rd,rs=ridge_select('delta_ridge',x,dy)
  clinical,_=ridge_select('history_only',np.column_stack([y0,dt]),y1)
  current,_=ridge_select('current_embedding_control',np.column_stack([e1,y0,dt]),y1)
  levelhistory,_=ridge_select('level_history_train',np.column_stack([y0,p1,dt]),y1)
 scaler=StandardScaler().fit(x[tr]);xx=torch.tensor(scaler.transform(x),dtype=torch.float32,device='cuda')
 yy=torch.tensor(dy,dtype=torch.float32,device='cuda');ti=torch.tensor(np.flatnonzero(tr),device='cuda');vi=torch.tensor(np.flatnonzero(va),device='cuda')
 ensemble=[];states=[]
 for seed in [11,17,23]:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  net=nn.Sequential(nn.Linear(x.shape[1],256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,1)).cuda()
  opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=.01)
  best=np.inf;wait=0;state=None
  for ep in range(100):
   net.train();order=ti[torch.randperm(len(ti),device='cuda')]
   for ix in order.split(256):
    loss=nn.functional.mse_loss(net(xx[ix]).squeeze(1),yy[ix]);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
   net.eval()
   with torch.no_grad():v=float((net(xx[vi]).squeeze(1)-yy[vi]).abs().mean())
   history.append(dict(model='delta_mlp',seed=seed,epoch=ep,val_mae=v))
   if v<best:best=v;state=copy.deepcopy(net.state_dict());wait=0
   else:wait+=1
   if wait>=12:break
  net.load_state_dict(state);net.eval()
  with torch.no_grad():ensemble.append(net(xx).squeeze(1).cpu().numpy())
  states.append({k:v.cpu() for k,v in state.items()});print('MLP',seed,'best validation MAE',best,'epochs',ep+1,flush=True)
 mlp=np.mean(ensemble,axis=0);ms=mae(dy[va],mlp[va]);selected='delta_mlp_ensemble' if ms<rs else 'delta_ridge';delta=mlp if ms<rs else rd
 print('SELECTED',selected,'val MAE',min(ms,rs),flush=True)
 base=np.column_stack([y0,p1,dt]);fuse=np.column_stack([base,delta])
 with threadpool_limits(limits=8):
  for name,X in [('fusion_without_delta',base),('fusion_with_delta',fuse)]:
   mod=make_pipeline(StandardScaler(),Ridge(alpha=1));mod.fit(X[va],y1[va]);saved[name]=mod
 preds={'carry_forward':y0,'level_only':p1,'anchored_level_difference':y0+p1-p0,'history_only':clinical,'level_history_train':levelhistory,'current_embedding_control':current,'direct_delta':y0+delta,'fusion_without_delta':saved['fusion_without_delta'].predict(base),'fusion_with_delta':saved['fusion_with_delta'].predict(fuse)}
 metrics=[]
 for name,p in preds.items():
  for split,mask in masks.items():
   for stratum,m in [('all',mask),('absolute_change_ge_0.5',mask&(np.abs(dy)>=.5))]:
    metrics.append(dict(model=name,split=split,stratum=stratum,n=int(m.sum()),patients=int(d.loc[m,'subject_id'].nunique()),mae=mae(y1[m],p[m]),rmse=float(np.sqrt(np.mean((y1[m]-p[m])**2))),level_r=corr(y1[m],p[m]),anchored_delta_r=corr(dy[m],p[m]-y0[m])))
 out=d.copy()
 for name,p in preds.items():out[name]=p
 out['delta_ridge_raw']=rd;out['delta_mlp_raw']=mlp
 out.to_parquet(O/'predictions.parquet',index=False)
 test=out.loc[te];comparisons=[]
 for other in preds:
  if other=='fusion_with_delta':continue
  err=np.abs(test.fusion_with_delta-test.y_post)-np.abs(test[other]-test.y_post)
  g=pd.DataFrame({'pid':test.subject_id,'diff':err}).groupby('pid')['diff'].agg(['sum','count'])
  rr=np.random.default_rng(20260909);bs=[]
  for _ in range(2000):
   ix=rr.integers(0,len(g),len(g));bs.append(float(g['sum'].to_numpy()[ix].sum()/g['count'].to_numpy()[ix].sum()))
  lo,hi=np.quantile(bs,[.025,.975]);comparisons.append(dict(comparison='fusion_with_delta minus '+other,mae_difference=float(err.mean()),ci_low=float(lo),ci_high=float(hi)))
 pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.DataFrame(history).to_csv(O/'selection_history.csv',index=False);pd.DataFrame(comparisons).to_csv(O/'paired_comparisons.csv',index=False)
 joblib.dump({'models':saved,'delta_scaler':scaler},O/'models.joblib');torch.save(states,O/'delta_mlp_states.pt')
 result=dict(experiment='exploratory frozen-embedding direct-delta learning',counts=counts,excluded=original-len(d),selection=selected,delta_validation_mae=dict(ridge=rs,mlp_ensemble=ms),comparisons=comparisons)
 (O/'results.json').write_text(json.dumps(result,indent=2))
 tab=pd.DataFrame(metrics).query("split=='test' and stratum=='all'")
 print(tab.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))
 lines=['# Direct delta-learning experiment','', '**Exploratory internal evaluation; frozen level-model embeddings.** This is not end-to-end waveform delta training or independent confirmation.', '',f'Patient-disjoint sets: {counts}. Excluded {original-len(d)} pairs for chronology. Selected using validation only: {selected}.', '', '| Model | Test MAE (mEq/L) | RMSE | Anchored change r |','|---|---|---|---|']
 for row in tab.to_dict('records'):
  r=row['anchored_delta_r'];rt=f'{r:.3f}' if pd.notna(r) else 'undefined (constant zero change)'
  lines.append(f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {rt} |")
 lines+=['','MAE of current K equals MAE of change when change is defined as prediction minus measured previous K; these are not independent endpoints. For the level-only model this anchored change differs from the difference of two level predictions.','', '| Fusion minus comparator | MAE difference | 95% paired patient-bootstrap CI |','|---|---|---|']
 for row in comparisons:lines.append(f"| {row['comparison']} | {row['mae_difference']:.4f} | {row['ci_low']:.4f} to {row['ci_high']:.4f} |")
 lines+=['','Negative MAE differences favour fusion. Primary contrast is fusion_without_delta, fitted on identical validation patients. The current-embedding control is trained on training patients and tuned on validation patients; it tests whether a second ECG is needed beyond previous K and the current ECG representation. All comparisons are exploratory; intervals condition on the fitted models.','','Previous K uses collection time as availability proxy; result-release timestamps were not verified. Labels are nearest within +/-60 minutes, and do not establish exact contemporaneous K. This cohort was previously analysed in the audit; the new test split is not an untouched independent cohort.','','Files: protocol.md, pairs.parquet, predictions.parquet, metrics.csv, paired_comparisons.csv, selection_history.csv, results.json, models.joblib, delta_mlp_states.pt.']
 (O/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
