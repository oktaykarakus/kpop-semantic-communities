from pathlib import Path
import os
OUT=Path(os.environ.get('KPOP_TASK_C_OUTPUT_DIR', Path(__file__).resolve().parent)).expanduser().resolve()
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np, pandas as pd, json, itertools, hashlib, importlib.metadata as md, sys
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari, mutual_info_score
from sklearn.metrics.cluster import entropy

def evaluate_NVI(index_pair, partitions):
    mutual_information = mutual_info_score(partitions[index_pair[0]], partitions[index_pair[1]])
    joint_entropy = (
        entropy(partitions[index_pair[0]])
        + entropy(partitions[index_pair[1]])
        - mutual_information
    )
    if abs(joint_entropy) < 1e-8:
        return 0.0
    return (joint_entropy - mutual_information) / joint_entropy

def write_csv(name,rows):
    with (OUT/name).open('x') as f:pd.DataFrame(rows).to_csv(f,index=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sizes(x):
    s=np.unique(x,return_counts=True)[1]
    return dict(n_communities=len(s),min_size=int(s.min()),q25_size=float(np.quantile(s,.25)),median_size=float(np.median(s)),q75_size=float(np.quantile(s,.75)),max_size=int(s.max()),singleton_communities=int((s==1).sum()))
def main():
    labs=np.load(OUT/'scan_labels.npy'); base=np.load(OUT/'baseline_labels.npy'); ids=np.load(OUT/'song_ids.npy')
    diag=json.loads((OUT/'scan_diagnostics.json').read_text()); rows=diag['scales']; ts=np.array([r['markov_time'] for r in rows]); ks=np.array([r['n_communities'] for r in rows]); nv=np.array([r['mean_pairwise_NVI'] for r in rows])
    assert labs.shape==(13,7891) and base.shape==(20,7891) and len(set(ids))==7891
    adj=np.array([evaluate_NVI((i,i+1),labs) for i in range(12)])
    for i,r in enumerate(rows):
        r['previous_scale_NVI']=float(adj[i-1]) if i else None
        r['next_scale_NVI']=float(adj[i]) if i<12 else None
    blocks=[]; start=0
    for i in range(1,14):
        if i==13 or ks[i]!=ks[start] or adj[i-1]>.15 or nv[i]>.15 or nv[i-1]>.15:
            if i-start>=3 and ks[start]>1 and np.max(nv[start:i])<=.15:blocks.append(list(range(start,i)))
            start=i
    med=float(np.median([len(np.unique(b)) for b in base])); lo=int(np.ceil(med/2)); hi=int(np.floor(med*2))
    relevant=[b for b in blocks if lo<=ks[b[0]]<=hi]
    if relevant:
        block=min(relevant,key=lambda b:(-(np.log10(ts[b[-1]])-np.log10(ts[b[0]])),float(nv[b].mean()),b[0])); chosen=block[len(block)//2]
    else:
        cand=[i for i,k in enumerate(ks) if lo<=k<=hi]
        if not cand:cand=[i for i,k in enumerate(ks) if 1<k<7891]
        chosen=min(cand,key=lambda i:(nv[i]+np.mean(adj[max(0,i-1):min(12,i+1)]),i))
    selected=labs[chosen]
    pairmat=np.array([[nmi(a,b) for b in base] for a in base]); medoid=int(np.argmax((pairmat.sum(1)-1)/19))
    comparisons=[]
    for seed,b in enumerate(base): comparisons.append(dict(comparison='alternative_vs_louvain',alternative_scale_index=chosen,markov_time=ts[chosen],louvain_seed=seed,NMI=nmi(selected,b),ARI=ari(selected,b),alternative_n_communities=len(np.unique(selected)),louvain_n_communities=len(np.unique(b)),overlap_reference=seed==medoid))
    for a,b in itertools.combinations(range(20),2): comparisons.append(dict(comparison='louvain_seed_pair',louvain_seed=a,louvain_seed_b=b,NMI=pairmat[a,b],ARI=ari(base[a],base[b])))
    write_csv('partition_comparison.csv',comparisons)
    write_csv('alternative_partition.csv',dict(song_id=ids,alternative_community=selected,markov_time=np.repeat(ts[chosen],7891),scale_index=np.repeat(chosen,7891)))
    overlap=[]; ref=base[medoid]
    for ac in np.unique(selected):
        an=int((selected==ac).sum())
        for bc in np.unique(ref):
            bn=int((ref==bc).sum()); count=int(((selected==ac)&(ref==bc)).sum())
            if count:overlap.append(dict(alternative_community=int(ac),louvain_seed=medoid,louvain_community=int(bc),n_overlap=count,alternative_size=an,louvain_size=bn,fraction_of_alternative=count/an,fraction_of_louvain=count/bn,jaccard=count/(an+bn-count)))
    write_csv('community_overlap.csv',overlap)
    assert sum(r['n_overlap'] for r in overlap)==7891
    write_csv('scale_summary.csv',rows)
    al= pd.DataFrame(comparisons[:20]); bl=pd.DataFrame(comparisons[20:])
    result=dict(selected_scale_index=chosen,markov_time=float(ts[chosen]),relevant_count_bounds=[lo,hi],robust_plateaus=[dict(indices=b,t_min=float(ts[b[0]]),t_max=float(ts[b[-1]]),count=int(ks[b[0]]),mean_NVI=float(nv[b].mean())) for b in blocks],has_relevant_plateau=bool(relevant),alternative_sizes=sizes(selected),louvain_medoid_seed=medoid,louvain_medoid_sizes=sizes(ref),comparison_NMI=al.NMI.describe().to_dict(),comparison_ARI=al.ARI.describe().to_dict(),baseline_NMI=bl.NMI.describe().to_dict(),baseline_ARI=bl.ARI.describe().to_dict(),selected_trial_NVI=float(nv[chosen]),runtime_seconds=diag['total_seconds'])
    with (OUT/'comparison_details.json').open('x') as f:json.dump(result,f,indent=2)
    versions = {
        "python": sys.version.split()[0],
        **{
            name: md.version(distribution)
            for name, distribution in {
                "pygenstability": "pygenstability",
                "numpy": "numpy",
                "scipy": "scipy",
                "pandas": "pandas",
                "scikit-learn": "scikit-learn",
            }.items()
        },
    }
    with (OUT/'package_versions.json').open('x') as f:json.dump(versions,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
