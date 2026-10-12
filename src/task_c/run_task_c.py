from pathlib import Path
import os
OUT=Path(os.environ.get('KPOP_TASK_C_OUTPUT_DIR', Path(__file__).resolve().parent)).expanduser().resolve()
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='1'
import numpy as np, pandas as pd, scipy, scipy.linalg as la, scipy.sparse as sp
import time, json, itertools, gc
from pygenstability import generalized_louvain
from pygenstability.pygenstability import evaluate_NVI, _assign_increasing_ids
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

def main():
    start=time.time()
    assert not (OUT/'scan_labels.npy').exists() and not (OUT/'alternative_method_runs.csv').exists()
    a=sp.load_npz(OUT/'graph.npz'); n=a.shape[0]
    d=np.asarray(a.sum(axis=1)).ravel(); pi=d/d.sum(); root=np.sqrt(d)
    # Exact normalized continuous-time walk, including diagonal adjacency weights.
    L=np.eye(n)-(sp.diags(1/root)@a@sp.diags(1/root)).toarray()
    print('Computing full diffusion spectrum once',flush=True)
    ev,v=la.eigh(L,overwrite_a=True,check_finite=False,driver='evd'); del L
    assert ev.min()>-1e-10
    ev=np.maximum(ev,0)
    print(f'Spectrum ready: {time.time()-start:.1f}s',flush=True)
    scales=np.logspace(-1,2,13)
    runs=[]; best=[]; tries=[]; checks=[]
    null=np.asarray([pi,pi])
    for si,t in enumerate(scales):
        st=time.time()
        # Pi exp(-t (I-D^-1 A)) = sqrt(D) exp(-t Lsym) sqrt(D) / sum(d).
        flow=(v*np.exp(-t*ev))@v.T
        flow*=root[:,None]; flow*=root[None,:]; flow/=d.sum()
        rowerr=float(np.max(np.abs(flow.sum(1)-pi)))
        asym=float(np.max(np.abs(flow-flow.T)))
        assert rowerr<1e-10 and asym<1e-12 and flow.min()>-1e-12
        # Same relative numerical cutoff convention as package (1e-8 of max).
        cutoff=1e-8*float(flow.max())
        flow[flow<cutoff]=0
        massloss=float(1-flow.sum())
        q=sp.tril(sp.csr_matrix(flow),format='coo'); del flow
        rows=q.row.astype(np.int64); cols=q.col.astype(np.int64); values=q.data
        labs=[]; scores=[]
        for trial in range(10):
            score,lab=generalized_louvain.run_louvain(rows,cols,values,len(values),null,2,1.0)
            lab=_assign_increasing_ids(np.asarray(lab)); labs.append(lab); scores.append(float(score))
            counts=np.unique(lab,return_counts=True)[1]
            runs.append(dict(scale_index=si,markov_time=t,trial=trial,stability=score,n_communities=len(counts),min_size=counts.min(),median_size=np.median(counts),max_size=counts.max(),selected_at_scale=False))
        bi=int(np.argmax(scores)); best.append(labs[bi]); tries.append(labs)
        runs[-10+bi]['selected_at_scale']=True
        nvis=[evaluate_NVI((i,j),labs) for i,j in itertools.combinations(range(10),2)]
        check=dict(scale_index=si,markov_time=float(t),n_communities=len(np.unique(labs[bi])),mean_pairwise_NVI=float(np.mean(nvis)),max_pairwise_NVI=float(np.max(nvis)),row_sum_error=rowerr,symmetry_error=asym,threshold_mass_loss=massloss,seconds=time.time()-st)
        checks.append(check)
        print(json.dumps(check),flush=True)
        del q,rows,cols,values; gc.collect()
    np.save(OUT/'scan_labels.npy',np.array(best),allow_pickle=False)
    np.save(OUT/'scan_all_trials.npy',np.array(tries),allow_pickle=False)
    with (OUT/'alternative_method_runs.csv').open('x') as f: pd.DataFrame(runs).to_csv(f,index=False)
    with (OUT/'scan_diagnostics.json').open('x') as f:json.dump(dict(scales=checks,total_seconds=time.time()-start),f,indent=2)
if __name__=='__main__': main()
