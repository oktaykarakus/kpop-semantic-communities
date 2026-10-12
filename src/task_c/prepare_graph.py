from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: os.environ[k]='1'
import numpy as np, pandas as pd, networkx as nx, community, json, hashlib, time
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
OUT=Path(os.environ.get('KPOP_TASK_C_OUTPUT_DIR', Path(__file__).resolve().parent)).expanduser().resolve()
ROOT=Path(os.environ.get('KPOP_PROJECT_ROOT', Path(__file__).resolve().parents[4])).expanduser().resolve()
B=Path(os.environ.get('KPOP_TASK_B_OUTPUT_DIR', ROOT/'revision/Rev2_TaskB_representation_comparison')).expanduser().resolve()
D=ROOT/'data/processed_revision1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'graph.npz').exists()
    paths=[D/'lyrics_embeddings_whole_song_truncated.npy',D/'lyrics_embedding_ids_whole_song_truncated.npy']
    ver=json.loads((B/'verification.json').read_text())
    hashes={str(p):sha(p) for p in paths}
    assert all(h==ver['input_sha256'][p] for p,h in hashes.items())
    x=np.load(paths[0]).astype(np.float32); ids=np.load(paths[1],allow_pickle=True).astype(str)
    assert x.shape==(7891,384) and len(set(ids))==7891 and np.isfinite(x).all()
    dist,idx=NearestNeighbors(n_neighbors=31,metric='cosine',n_jobs=1).fit(x).kneighbors(x)
    g=nx.Graph()
    for i,u in enumerate(ids):
        for j,d in zip(idx[i,1:],dist[i,1:]): g.add_edge(u,ids[int(j)],weight=1.0-float(d))
    g.add_nodes_from(ids.tolist())
    assert len(g)==7891 and g.number_of_edges()==209805 and nx.number_of_selfloops(g)==14 and nx.is_connected(g)
    sparse.save_npz(OUT/'graph.npz',sparse.csr_matrix(nx.to_scipy_sparse_array(g,nodelist=ids,weight='weight',format='csr')))
    np.save(OUT/'song_ids.npy',ids,allow_pickle=False)
    baseline=pd.read_csv(B/'louvain_repeated_runs.csv'); baseline=baseline[baseline.representation=='whole_song_truncated'].set_index('seed')
    labels=[]
    for seed in range(20):
        p=community.best_partition(g,weight='weight',resolution=2.0,random_state=seed)
        lab=np.array([p[u] for u in ids]); labels.append(lab)
        q=community.modularity(p,g,weight='weight')
        assert len(set(lab))==baseline.loc[seed,'n_communities'] and abs(q-baseline.loc[seed,'modularity'])<1e-12
        print(f'Baseline seed {seed}: exact Task B count/modularity reproduced',flush=True)
    np.save(OUT/'baseline_labels.npy',np.array(labels),allow_pickle=False)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    details=dict(input_sha256=hashes,nodes=len(g),edges=g.number_of_edges(),self_loops=14,baseline_verified_seeds=list(range(20)),graph_sha256=sha(OUT/'graph.npz'),input_unchanged=True)
    with (OUT/'graph_verification.json').open('x') as f: json.dump(details,f,indent=2)
if __name__=='__main__': main()
