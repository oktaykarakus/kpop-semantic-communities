from pathlib import Path
import os
import json, hashlib
import pandas as pd
import numpy as np
P=Path(os.environ.get('KPOP_TASK_B_OUTPUT_DIR', Path(__file__).resolve().parent/'outputs')).expanduser().resolve()
summary='''# Rev2 Task B — representation decision

**Recommendation: SWITCH TO WHOLE-SONG TRUNCATED.**

On this bounded, fixed-parameter comparison, line-mean has a modularity advantage but no overall empirical advantage sufficient to justify retaining it. Whole-song truncated has substantially greater neighbour-versus-random separation and better repeated-partition stability. This recommendation selects a representation; it does not establish semantic validity or implement the switch.

## Evidence

1. **Raw cosine is misleading here.** Line-mean neighbour mean is 0.9784, but its random baseline is already 0.9395. Whole-song truncated gives 0.7836 versus 0.5420. The respective cosine gaps are **0.03885** (95% bootstrap CI 0.03838–0.03931) and **0.24162** (0.24087–0.24240).
2. **The difference survives standardization.** Pooled-SD separation is **0.690** (0.668–0.711) for line-mean versus **1.995** (1.978–2.011) for whole-song truncated. The paired whole-minus-line difference is **1.305**, CI **1.280–1.330**. The raw gap difference is 0.20277, CI 0.20184–0.20364. These are descriptive effect sizes, not claims based on large-sample p-values.
3. **Line-mean wins modularity; whole-song wins repeatability.** Across seeds 0–19, line-mean modularity is **0.46498 ± 0.00669**, median 0.46620, range 0.44727–0.47267. Whole-song is **0.41523 ± 0.00338**, median 0.41450, range 0.41008–0.42092. Mean pairwise NMI/ARI are **0.5779/0.4326** versus **0.6226/0.5027**, respectively. Community counts average 17.6 (16–22) versus 20.8 (19–22). Modularity across different graphs is not a semantic-quality score, and its advantage alone does not outweigh the separation and repeatability evidence.
4. **No fragmentation penalty, but topology differs.** Both graphs contain all 7,891 songs in one connected component. Line-mean has 217,868 edges and mean degree 55.22; whole-song has 209,805 edges and mean degree 53.18 (3.7% fewer edges). Whole-song has a larger maximum degree (539 versus 405). Canonical tied-neighbour handling leaves 6 versus 14 self-loops; these are retained in graph analysis and excluded from the neighbour-distribution summaries.
5. **The other two representations provide bounded context.** Unique-line-mean has standardized separation 0.592, mean modularity 0.46872, NMI 0.6252 and ARI 0.4906. Chunked-document has 1.931, 0.44927, 0.6195 and 0.5310, respectively. Whole-song has the strongest separation of the four, though not the highest modularity or every stability measure. No broader representation search is warranted by this task.

## Scope and caveats

All four inputs passed shape, finite/nonzero-vector and exact ID-order/set checks before analysis. The same k=30, canonical weighted undirected graph construction, Louvain resolution=2.0, 20 seeds, 236,730 random-pair draws and 1,000 anchor bootstrap resamples were used throughout. Reported modularity is standard weighted modularity (evaluation gamma=1), matching WP2. SDs are sample SDs. Stability uses all 190 seed pairs.

Intervals condition on this corpus, existing embeddings, fixed neighbour sets and sampled random partners. Anchor resampling preserves within-song dependence but does not remove shared-target or artist dependence. No graph-refitting or generalization interval is claimed. Separation can reflect script, form or embedding geometry; neither separation nor modularity proves thematic validity. Whole-song truncation may omit later lyrics; this comparison does not quantify that content loss. The recommendation is a transparent multi-metric judgment, not a preregistered composite score or proof of universal superiority.

The rebuilt canonical line-mean edge set exactly matches the saved 217,868-edge graph; maximum weight difference is 1.79e-7. Seed 9 reproduces modularity 0.4726671131713442. This supports pipeline comparability despite floating-point differences.

**Stop Task B here.** Keep Rev1 and Tasks A/A2 as historical results. Their line-mean community labels, characterizations and examples cannot automatically be attributed to a future whole-song partition. Any later adoption must reconcile those downstream claims separately. No manuscript edits, new embeddings, Task A/A2 reruns, alternative community methods or canonical-file replacements were performed.

## Files and reproduction

The five requested CSVs contain four representation summaries, eight similarity distributions, ten effect/contrast rows, 80 Louvain runs, and the direct line-versus-whole decision table. `provenance.md` and `verification.json` document inputs, hashes, definitions, sampling and versions. `run_task_b.py` creates the numeric outputs and provenance; `finish_task_b.py` validates them and writes this report and `output_validation.json`. Both refuse to overwrite their outputs. To reproduce, use a fresh copy of the scripts with OUT/P changed to a new empty Task B destination and run them with the recorded Python interpreter. The report text is the interpretation of this run and must be reconsidered if inputs or methods change.
'''
expected={'representation_summary.csv':4,'similarity_dispersion.csv':8,'separation_effects.csv':10,'louvain_repeated_runs.csv':80,'line_mean_vs_whole_song_decision.csv':20}
checks={}
for name,n in expected.items():
    df=pd.read_csv(P/name)
    assert len(df)==n,(name,len(df))
    assert not df.isna().any().any(),name
    assert np.isfinite(df.select_dtypes('number').to_numpy()).all(),name
    checks[name]={'rows':len(df),'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}
runs=pd.read_csv(P/'louvain_repeated_runs.csv')
for rep,g in runs.groupby('representation'):
    assert sorted(g.seed.tolist())==list(range(20))
    assert (g.k==30).all() and (g.resolution==2).all()
assert (pd.read_csv(P/'similarity_dispersion.csv').n==236730).all()
effects=pd.read_csv(P/'separation_effects.csv')
assert (effects.ci95_low<=effects.estimate).all() and (effects.estimate<=effects.ci95_high).all()
v=json.loads((P/'verification.json').read_text())
assert v['canonical_graph']['canonical_edge_set_equal']
assert abs(v['canonical_graph']['seed9_modularity']-.4726671131713442)<1e-10
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in v['input_sha256'].items())
with (P/'TaskB_summary.md').open('x') as f: f.write(summary)
checks['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
checks['all_checks_passed']=True
with (P/'output_validation.json').open('x') as f: json.dump(checks,f,indent=2)
print('Output validation passed. Recommendation: SWITCH TO WHOLE-SONG TRUNCATED.')
