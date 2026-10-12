from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import sys, json, hashlib, time, platform
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
import community
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT=Path(os.environ.get('KPOP_PROJECT_ROOT', Path(__file__).resolve().parents[4])).expanduser().resolve()
OUT=Path(os.environ.get('KPOP_TASK_B_OUTPUT_DIR', Path(__file__).resolve().parent/'outputs')).expanduser().resolve()
DATA=ROOT/'data/processed_revision1'
REPS=['line_mean','unique_line_mean','chunked_document','whole_song_truncated']
N=7891; K=30; B=1000; SAMPLE_SEED=20261006; BOOT_SEED=20261007

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):
    v=np.asarray(v,dtype=float).ravel()
    q=np.quantile(v,[.01,.05,.25,.5,.75,.95,.99])
    return dict(n=len(v),mean=v.mean(),sd=v.std(ddof=1),p01=q[0],p05=q[1],q1=q[2],median=q[3],q3=q[4],iqr=q[4]-q[2],p95=q[5],p99=q[6],minimum=v.min(),maximum=v.max())
def write_csv(rows,name):
    with (OUT/name).open('x') as f: pd.DataFrame(rows).to_csv(f,index=False)
def write_text(name,text):
    with (OUT/name).open('x') as f: f.write(text)
def run(rep):
    t=time.time()
    x=np.load(DATA/f'lyrics_embeddings_{rep}.npy').astype(np.float32)
    ids=np.load(DATA/f'lyrics_embedding_ids_{rep}.npy',allow_pickle=True).astype(str)
    nn=NearestNeighbors(n_neighbors=31,metric='cosine',n_jobs=1).fit(x)
    dist,idx=nn.kneighbors(x)
    # Preserve canonical WP2 graph insertion order and finalise_k30_seed9 weights.
    g=nx.Graph()
    for i,uid in enumerate(ids):
        for j,d in zip(idx[i,1:],dist[i,1:]): g.add_edge(uid,ids[int(j)],weight=1.0-float(d))
    g.add_nodes_from(ids.tolist())
    # Distribution excludes self by identity; ties otherwise retain sklearn ordering.
    near=np.asarray([1.0-dist[i,idx[i]!=i][:K].astype(float) for i in range(N)])
    rng=np.random.default_rng(SAMPLE_SEED)
    partners=rng.integers(0,N-1,size=(N,K))
    partners+=partners>=np.arange(N)[:,None]
    z=normalize(x)
    random=np.empty((N,K))
    for j in range(K): random[:,j]=np.einsum('ij,ij->i',z,z[partners[:,j]])
    means=np.column_stack([near.mean(1),random.mean(1),(near**2).mean(1),(random**2).mean(1)])
    def effects(m):
        a,b,a2,b2=m
        gap=a-b
        sd=np.sqrt(((a2-a*a)+(b2-b*b))/2 * (N*K)/(N*K-1))
        return np.array([gap,gap/sd])
    point=effects(means.mean(0))
    rng=np.random.default_rng(BOOT_SEED)
    boots=np.asarray([effects(means[rng.integers(0,N,size=N)].mean(0)) for _ in range(B)])
    effects_rows=[]
    for j,name in enumerate(['mean_cosine_gap','pooled_sd_standardized_gap']):
        lo,hi=np.quantile(boots[:,j],[.025,.975])
        effects_rows.append(dict(representation=rep,effect=name,estimate=point[j],ci95_low=lo,ci95_high=hi,bootstrap_replicates=B,bootstrap_unit='anchor_song',random_pairs=N*K))
    runs=[]; labels=[]
    for seed in range(20):
        p=community.best_partition(g,weight='weight',resolution=2.0,random_state=seed)
        lab=np.asarray([p[u] for u in ids]); labels.append(lab)
        runs.append(dict(representation=rep,k=K,resolution=2.0,seed=seed,modularity=community.modularity(p,g,weight='weight'),n_communities=len(set(lab))))
        if seed%5==4: print(f'{rep}: {seed+1}/20 seeds ({time.time()-t:.0f}s)',flush=True)
    nmi=[]; ari=[]
    for a,b in combinations(labels,2):
        nmi.append(normalized_mutual_info_score(a,b)); ari.append(adjusted_rand_score(a,b))
    degree=np.array([d for _,d in g.degree()]); weights=np.array([v['weight'] for _,_,v in g.edges(data=True)])
    comps=list(nx.connected_components(g))
    row=dict(representation=rep,k=K,resolution=2.,n_nodes=len(g),n_edges=g.number_of_edges(),n_self_loops=nx.number_of_selfloops(g),density=nx.density(g),mean_degree=degree.mean(),median_degree=np.median(degree),min_degree=degree.min(),max_degree=degree.max(),n_connected_components=len(comps),largest_component_fraction=max(map(len,comps))/N,mean_edge_weight=weights.mean(),min_edge_weight=weights.min(),max_edge_weight=weights.max(),nonpositive_edges=int((weights<=0).sum()),canonical_knn_self_slots=int((idx[:,1:]==np.arange(N)[:,None]).sum()),knn_mean=near.mean(),random_mean=random.mean(),mean_cosine_gap=point[0],pooled_sd_standardized_gap=point[1],n_louvain_runs=20,n_seed_pairs=190)
    for key,values in [('modularity',[r['modularity'] for r in runs]),('n_communities',[r['n_communities'] for r in runs]),('pairwise_NMI',nmi),('pairwise_ARI',ari)]:
        s=stats(values)
        for field in ['mean','sd','median','minimum','maximum']: row[f'{key}_{field}']=s[field]
    verification={}
    if rep=='line_mean':
        old=pd.read_parquet(DATA/'lyric_graph_revision1.parquet')
        oldedges={(min(a,b),max(a,b)):float(w) for a,b,w in old[['source','target','weight']].itertuples(index=False,name=None)}
        newedges={(min(a,b),max(a,b)):v['weight'] for a,b,v in g.edges(data=True)}
        verification={'canonical_edge_set_equal':oldedges.keys()==newedges.keys(),'saved_edge_count':len(oldedges),'rebuilt_edge_count':len(newedges),'max_common_edge_weight_difference':max(abs(w-newedges[e]) for e,w in oldedges.items() if e in newedges),'seed9_modularity':runs[9]['modularity']}
        print('Canonical verification:',verification,flush=True)
    return row,[dict(representation=rep,distribution=label,**stats(v)) for label,v in [('knn_nonself',near),('random_nonself',random)]],effects_rows,runs,boots,verification

def main():
    started=time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    names=['representation_summary.csv','similarity_dispersion.csv','separation_effects.csv','louvain_repeated_runs.csv','line_mean_vs_whole_song_decision.csv','TaskB_summary.md','provenance.md','verification.json']
    assert not any((OUT/n).exists() for n in names),'Refusing to overwrite outputs'
    inputs=[]; base=None
    for rep in REPS:
        xp=DATA/f'lyrics_embeddings_{rep}.npy'; ip=DATA/f'lyrics_embedding_ids_{rep}.npy'
        x=np.load(xp); ids=np.load(ip,allow_pickle=True).astype(str)
        if base is None: base=ids
        assert x.shape==(N,384) and ids.shape==(N,) and len(set(ids))==N and np.array_equal(ids,base),'STOP: input shape/alignment failed'
        assert np.isfinite(x).all() and (np.linalg.norm(x,axis=1)>0).all()
        inputs.extend([xp,ip])
    inputs += [DATA/'lyric_graph_revision1.parquet', ROOT/'revision/WP2_revision1_graph_robustness/run_wp2_revision1.py',ROOT/'revision/WP2_revision1_graph_robustness/finalise_k30_seed9.py',ROOT/'revision/WP2_revision1_graph_robustness/config.py']
    hashes={str(p):sha(p) for p in inputs}
    print('Verified all four aligned arrays; starting comparison.',flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool: results=list(pool.map(run,REPS))
    assert all(sha(Path(p))==h for p,h in hashes.items()),'Inputs changed during run'
    rows=[r[0] for r in results]; effects_rows=sum([r[2] for r in results],[])
    a=rows[0]; b=rows[3]
    delta=results[0][4]-results[3][4]
    for j,key in enumerate(['mean_cosine_gap','pooled_sd_standardized_gap']):
        lo,hi=np.quantile(delta[:,j],[.025,.975])
        effects_rows.append(dict(representation='line_mean_minus_whole_song_truncated',effect=key,estimate=a[key]-b[key],ci95_low=lo,ci95_high=hi,bootstrap_replicates=B,bootstrap_unit='paired_anchor_song',random_pairs=N*K))
    write_csv(rows,names[0]); write_csv(sum([r[1] for r in results],[]),names[1]); write_csv(effects_rows,names[2]); write_csv(sum([r[3] for r in results],[]),names[3])
    keys=['knn_mean','random_mean','mean_cosine_gap','pooled_sd_standardized_gap','modularity_mean','modularity_sd','modularity_median','modularity_minimum','modularity_maximum','pairwise_NMI_mean','pairwise_ARI_mean','n_communities_mean','n_communities_minimum','n_communities_maximum','n_edges','mean_degree','max_degree','n_self_loops','n_connected_components','largest_component_fraction']
    write_csv([dict(metric=k,line_mean=a[k],whole_song_truncated=b[k],line_mean_minus_whole_song=a[k]-b[k]) for k in keys],names[4])
    verification=dict(alignment='All four (7891,384); 7891 unique string IDs; exact identical order and set; finite nonzero vectors',canonical_graph=results[0][5],input_sha256=hashes,inputs_unchanged_after_run=True,versions={m.__name__:m.__version__ for m in [np,pd,nx,community,sklearn]},python=sys.executable,runtime_seconds=time.time()-started)
    write_text('verification.json',json.dumps(verification,indent=2))
    write_text('provenance.md',f'''# Task B provenance

Script: `{OUT/'run_task_b.py'}` (SHA-256: {sha(Path(__file__))}). Executed with `{sys.executable}`; Python {platform.python_version()}. Package versions and exact input paths/SHA-256 hashes: `verification.json`. Runtime: {time.time()-started:.1f} seconds. No original code patched or executed; standalone wrapper reproduces the inspected canonical graph and Louvain functions. Four independent representations run in four worker processes, single-thread numerical libraries.

All four embeddings verified before analysis: 7891 x 384, finite nonzero rows, 7891 unique IDs and exact common ID order/set. No realignment, filtering, embedding generation or parameter tuning. Input hashes verified unchanged afterwards. Only new files in this Task B directory are written using exclusive creation; rerun refuses existing outputs. No manuscript, Task A/A2 or excluded paper operations.

## Fixed definitions
k=30, cosine NearestNeighbors on float32 arrays, 31 returned slots. Canonical graph: drop first slot exactly as WP2/finalise_k30_seed9, undirected union, overwrite reciprocal edge weight on later insertion, weight=1-float(distance), preserve node/edge insertion order, append all IDs. Ties can leave self-loops, retained and counted. Graph density is NetworkX density including these loops; degree counts loops twice. Saved canonical line-mean graph is compared read-only; verification records edge equality and weight differences.

Similarity distributions use the 30 non-self neighbours selected by explicitly excluding the anchor ID among the 31 slots. Thus they avoid the canonical graph's tied-self artifact while preserving graph reproducibility. Random baseline: NumPy default_rng seed {SAMPLE_SEED}, N x 30 integers uniform on [0,N-2], shifted at the anchor index to exclude self. Exactly {N*K} ordered pair draws per representation, 30 per anchor, with replacement (repeated and reciprocal pairs allowed). Identical sampled IDs across representations; true neighbours are not excluded. Cosines use sklearn L2 normalization and float32 dot products; summaries in float64.

Dispersion n, mean, sample SD (ddof=1), median, Q1/Q3, IQR, 1/5/95/99 percentiles, min/max; NumPy default linear quantiles. Directed neighbour slots are counted, not unique graph edges. Effect sizes: mean cosine gap=kNN mean minus random mean; standardized gap=gap / sqrt((sample variance kNN + sample variance random)/2). This is a descriptive pooled-SD effect, not an independence-based test.

Bootstrap: {B} percentile resamples, seed {BOOT_SEED}, resample 7891 anchor rows with replacement, keeping each row's 30 neighbours and 30 random draws together. Recompute moments and both effects. The same bootstrap anchor indices across representations give paired line-minus-whole differences and 95% intervals. This conditions on existing embeddings/neighbourhoods and sampled random partners; it does not refit graphs or estimate artist/corpus generalization uncertainty. Shared targets and artist dependence remain. No p-values.

Louvain: python-louvain best_partition, weight='weight', resolution=2.0, random_state seeds 0..19. Reported modularity is python-louvain standard weighted modularity (evaluation gamma=1), matching WP2; it is NOT a resolution-2 objective value. Twenty runs each; sample SD, median, min/max; community count summaries. Stability uses all 190 unordered seed pairs: sklearn NMI (arithmetic averaging default) and ARI. Stability is optimizer repeatability, not semantic validity. Different graphs' modularities are descriptive and affected by topology. No universal threshold or significance claim is attached to modularity differences.

## Exact inputs and code references
'''+''.join(f'- `{p}` — SHA-256 `{h}`\n' for p,h in hashes.items()))
    print(pd.DataFrame(rows)[['representation','mean_cosine_gap','pooled_sd_standardized_gap','modularity_mean','pairwise_NMI_mean','pairwise_ARI_mean','n_edges','n_self_loops']].to_string(index=False),flush=True)
    print('Analysis complete; summary requires evidence-based decision.',flush=True)
if __name__=='__main__': main()
