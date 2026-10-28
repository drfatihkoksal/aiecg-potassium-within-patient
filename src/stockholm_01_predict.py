"""Frozen published Swedish potassium models on fixed MIMIC cohorts.
Primary: Zenodo model_3 (chosen by the authors); other seeds are sensitivity.
Outputs remain source-standardized. No training or output calibration here.
"""
import argparse, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / 'data/interim'
UPSTREAM = ROOT / 'checkpoint/ecg-electrolyte-regression'
sys.path.insert(0, str(UPSTREAM))
from src.models.resnet import ResNet1d

LEADS = ['DI','DII','DIII','AVR','AVF','AVL','V1','V2','V3','V4','V5','V6']
KEEP = [0,1,6,7,8,9,10,11]
order, _ = signal.ellipord(0.8/250, 0.2/250, 0.5, 40)
SOS = signal.iirfilter(order, 0.8/250, 0.5, 40, btype='high', ftype='ellip', output='sos')
NOTCH = signal.iirnotch(50, 30, fs=500)

def transform(x):
    # Matches the cited ecg-preprocessing library: filter at native rate first.
    x = np.asarray(x, dtype=np.float64)[KEEP]
    assert np.isfinite(x).all()
    y = signal.sosfiltfilt(SOS, x, padtype='constant', axis=-1)
    y = signal.filtfilt(*NOTCH, y, axis=-1)
    y = signal.resample_poly(y, up=400, down=500, axis=-1)
    return np.pad(y, ((0,0),(48,48))).astype(np.float32)

class Waves(Dataset):
    def __init__(self, rows, n):
        self.rows = rows
        self.raw = np.memmap(INTERIM.parent/'waveforms/ecg_raw_f16.npy', mode='r', dtype='float16', shape=(n,12,5000))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return torch.from_numpy(transform(self.raw[self.rows[i]]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--limit',type=int)
    a=ap.parse_args();torch.set_num_threads(4);torch.manual_seed(0)
    wi=pd.read_parquet(INTERIM/'wave_index.parquet')
    sp=pd.read_parquet(INTERIM/'split_potassium.parquet')
    audit=pd.read_parquet(INTERIM/'audit_potassium.parquet')
    ids=set(sp.loc[sp.split.isin(['train','test']),'study_id']) | set(audit.pre_study) | set(audit.post_study)
    idx=wi[wi.study_id.isin(ids)].sort_values('row').copy()
    assert len(idx)==len(ids) and idx.qc_pass.all()
    assert not set(sp.loc[sp.split=='train','subject_id']) & set(audit.subject_id)
    assert not set(sp.loc[sp.split=='test','subject_id']) & set(audit.subject_id)
    if a.limit:idx=idx.iloc[:a.limit]
    ds=Waves(idx.row.to_numpy(),len(wi))
    # Numerical verification against unchanged author preprocessing, independent signals.
    pp=ROOT/'checkpoint/ecg-preprocessing/ecgprep/preprocess.py'
    spec=importlib.util.spec_from_file_location('verified_author_preprocess',pp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    maxdiff=0.
    for i in [0,min(17,len(ds)-1),len(ds)-1]:
        x=np.asarray(ds.raw[ds.rows[i]],dtype=np.float64)
        ref,fs,leads=mod.preprocess_ecg(x,500,LEADS,new_freq=400,new_len=4096,remove_baseline=True,remove_powerline=50)
        np.testing.assert_allclose(transform(x),ref.astype('float32'),rtol=0,atol=0)
        assert leads==['DI','DII','V1','V2','V3','V4','V5','V6'] and fs==400
    models=[]
    for seed in range(5):
        folder=ROOT/f'checkpoint/stockholm/model_{seed}'
        cfg=json.loads((folder/'train_config.json').read_text())
        assert cfg['normalize'] and not cfg['use_metadata']
        state=torch.load(folder/'model.pth',map_location='cpu',weights_only=True)['model']
        widths=[state[f'resblock1d_{j}.conv1.weight'].shape[0] for j in range(5)]
        model=ResNet1d((8,4096),list(zip(widths,[4096,1024,256,64,16])),1,kernel_size=17,dropout_rate=0.5)
        model.load_state_dict(state,strict=True);model.cuda().eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        models.append(model)
    loader=DataLoader(ds,batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,shuffle=False)
    result=np.empty((len(ds),5),dtype=np.float32);start=time.time();n=0
    with torch.inference_mode():
        for x in loader:
            x=x.cuda(non_blocking=True)
            # Full float32 avoids introducing autocast differences into external audit.
            yp=torch.cat([m(x) for m in models],dim=1).cpu().numpy()
            assert np.isfinite(yp).all()
            result[n:n+len(x)]=yp;n+=len(x)
            if n % (a.batch_size*100)==0:print(f'{n:,}/{len(ds):,} ECGs; {time.time()-start:.0f}s',flush=True)
    output=idx[['subject_id','study_id','ecg_time','row']].reset_index(drop=True)
    for seed in range(5):output[f'stockholm_{seed}_z']=result[:,seed]
    name='pred_stockholm_smoke' if a.limit else 'pred_stockholm'
    output.to_parquet(INTERIM/f'{name}.parquet',index=False)
    provenance={'primary_model':3,'source_doi':'10.5281/zenodo.7456316','n_ecgs':len(output),'parameters_per_model':sum(p.numel() for p in models[0].parameters()),'output_units':'Swedish training-standardized potassium; original mean and SD unavailable','preprocessing':'mV; native 500Hz elliptic highpass 0.8Hz, stopband 0.2Hz, ripple .5dB, attenuation40dB, constant pad; 50Hz notch Q30; polyphase400Hz; center zero-pad4096; I II V1-V6','upstream_preprocessing_exact_float32_match':True,'weights_frozen':True,'precision':'float32','elapsed_seconds':time.time()-start,'git_model':subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip(),'git_preprocessing':subprocess.check_output(['git','-C',str(ROOT/'checkpoint/ecg-preprocessing'),'rev-parse','HEAD'],text=True).strip()}
    (ROOT/'outputs'/f'{name}_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2),flush=True)

if __name__=='__main__':main()
