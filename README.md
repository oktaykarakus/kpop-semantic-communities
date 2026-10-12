# Analytical Code for Semantic Communities and Boundary-Spanning Lyrics in K-pop

This code-only package contains eight analytical scripts associated with *Semantic Communities and Boundary-Spanning Lyrics in K-pop: A Graph-Based Unsupervised Analysis*, by Oktay Karakuş and Can Eyüpoğlu. Version `v1.0.0` is licensed under MIT, with Copyright (c) 2026 Oktay Karakuş. It is not a data release or a complete end-to-end reproduction archive.

## Scope and reproduction limits

The scripts cover the four-representation comparison and graph evidence (Task B), multiscale Markov Stability analysis (Task C), final community characterization and script-composition analysis (Task D), and participation, novelty, and artist-level analysis (Task E). They reuse frozen or restricted inputs created elsewhere in the project.

The released material is analytical code only. It does not include the historical inputs required to run the analyses: the canonical 7,891-song corpus, lyric records, embeddings, graph arrays, partitions, and intermediate results are unavailable in this package. The historical preparation pipeline from the upstream lyric archive to the canonical corpus, and the historical embedding-generation code, have not been independently reconstructed. The exact relationship between the inspected upstream snapshot and the study input is unverified. Execution depends on obtaining compatible inputs and historical dependencies separately. The package must not be described as providing full end-to-end reproducibility or an independent reproduction of the published results.

## Upstream lyric dataset

The project identifies EX3exp's public [`Kpop-lyric-datasets`](https://github.com/EX3exp/Kpop-lyric-datasets) repository as an upstream repository associated with the study's lyric records. The inspected snapshot is commit `a7d141f5431fffb61c52c8ec1e48553975fa6350` (2023-12-03); its tracked `melon/` tree contained 25,876 JSON files. The repository README separately reports 25,696 songs. That discrepancy and the exact snapshot-to-study-input relationship remain unresolved. The archive is not attributed to Melon as distributor merely because Melon's Monthly Chart Ranking Top 100 provides chart context.

To inspect the upstream repository, use its public GitHub instructions and verify the relevant terms and rights before use. For example:

```sh
git clone https://github.com/EX3exp/Kpop-lyric-datasets.git
cd Kpop-lyric-datasets
git checkout a7d141f5431fffb61c52c8ec1e48553975fa6350
```

This package does not download, contain, or redistribute lyrics. Repository access does not establish permission to redistribute lyric texts, and this package does not promise that the final 7,891-song corpus can be reconstructed from that snapshot.

## Package contents

- `src/task_b/`: representation-comparison analysis and its report finalizer.
- `src/task_c/`: graph preparation, Markov Stability scan, and partition comparison.
- `src/task_d/`: community characterization and output validation/report finalizer.
- `src/task_e/`: participation, novelty, and artist-level analysis.
- `environments/README.md`: recorded environment facts and inferred dependency needs, kept distinct.
- `provenance/`: original/staged script mapping, hashes, result-linkage evidence, and limitations.
- `results/approved-aggregate-only/`: intentionally empty; no result file has been approved for release.
- `.gitignore`: excludes known generated and potentially restricted task outputs from version control.
- `CITATION.cff`: citation metadata for `v1.0.0`.
- `LICENSE`: MIT licence; Copyright (c) 2026 Oktay Karakuş.

No datasets, lyric excerpts, embedding arrays, graph/partition arrays, song-level tables or identifiers, or generated analysis outputs are staged. The staged Task E narrative-annotation mapping is also removed; see the note below.

## Inputs, outputs, and historical dependencies

Inputs below are expected project artifacts, not included files. Paths are relative to the project root unless otherwise stated. Several outputs contain song identifiers or lyric-derived material and must remain private unless separately reviewed and approved.

| Task | Expected inputs | Outputs written by the scripts | Important dependency / limitation |
|---|---|---|---|
| **B** | Four aligned embedding arrays and ID arrays under `data/processed_revision1/`; the saved Rev1 graph parquet; and the WP2 Rev1 graph scripts/config under `revision/WP2_revision1_graph_robustness/`. | Representation summary, similarity dispersion, separation effects, repeated Louvain runs, representation decision table, verification/provenance, and a finalized report. | WP2 files and the saved graph are read for hashes and the canonical line-mean comparison; they are not run to calculate the four representation results. These historical verification dependencies are nevertheless required for the staged runner to complete as written. The scripts do not generate embeddings. |
| **C** | Whole-song-truncated embeddings and IDs under `data/processed_revision1/`; Task B `verification.json` and `louvain_repeated_runs.csv`; then Task C graph and scan intermediates. | `graph.npz`, IDs, baseline labels, scan labels/trials/diagnostics, partition comparisons, overlap and scale summaries. | Task C is a partial analytical component. PyGenStability uses a private helper and stochastic trials without a recorded seed parameter. The package does not include its required arrays or prove an independent rerun of the frozen scan. |
| **D** | Canonical `lyrics.parquet`; whole-song embeddings; Task B run summaries; Task C graph, IDs, and baseline labels; Task A/A2 materials used by the finalizer to validate prior evidence. | Final partition, community profiles/terms, representative songs, interpretation evidence, script summaries/association, metadata, validation and report. | Task A/A2 manifest and source artifacts are required by the finalizer only for integrity checks; they do not supply Task D's quantitative calculations. Outputs include song identifiers and lyric snippets. Do not release generated outputs without a separate field-level rights review. |
| **E** | Canonical corpus, whole-song embeddings/IDs, Task C graph/IDs, Task D partition/interpretation/evidence, and WP4 Rev1 code/configuration plus saved summary, metrics, and artist table. An optional narrative-annotation JSON may be supplied with `KPOP_TASK_E_ANNOTATIONS`. | Song- and artist-level metrics, example records, comparisons, correlations, provenance and validation. | Current Rev2 Task E metrics require corpus, embeddings, Task C graph and Task D partition/evidence. WP4 metrics and artist table supply the descriptive Rev1 comparison; WP4 script/config/summary are hash/provenance checks. The staged script requires those WP4 paths to complete as written, but the WP4 computational code is not the Rev2 implementation. |

The staged scripts use `KPOP_PROJECT_ROOT` where project-root resolution is needed. The inferred default assumes this package remains at its current location inside the project; a cloned or relocated release must set `KPOP_PROJECT_ROOT` to the authorized project data root. Set `KPOP_TASK_B_OUTPUT_DIR`, `KPOP_TASK_C_OUTPUT_DIR`, `KPOP_TASK_D_OUTPUT_DIR`, and `KPOP_TASK_E_OUTPUT_DIR` to private working directories before execution. The B/C/D directories are also consumed as inputs by downstream tasks; use the same directory value throughout the workflow. Task B and Task D create their output directories; Task C creates its directory during graph preparation. Task E creates its output directory. Outputs from B–E can contain restricted arrays, identifiers, excerpts, or song-level tables. The staged `.gitignore` excludes known generated filenames when default output directories are used, but is not a substitute for private output locations or manual release review.

### Task E narrative-example annotations

For data minimization, the staged Task E script does not contain the five original `song_uid`-keyed narrative annotations. Its quantitative calculations, thresholds, comparisons, ranking, and example-selection procedure are unchanged. If an authorized private mapping is available, provide it as a UTF-8 JSON object of at most five string `song_uid` keys to string paraphrase values and set `KPOP_TASK_E_ANNOTATIONS` to a file stored outside this release package. The loader rejects a mapping located inside the package; its SHA-256 is recorded in Task E run validation when supplied. If it is not supplied, only those five illustrative narrative annotations are omitted; the quantitative Task E outputs are unaffected. No replacement identifiers or annotations are invented. The mapping itself is not staged. Any outputs produced with a mapping must remain private unless separately reviewed.

## Historical dependency disposition

| Dependency | Role in the staged script | Release recommendation |
|---|---|---|
| **WP2 Rev1 scripts/config and saved Rev1 graph (Task B)** | Used for input hashing and a saved-graph comparison/verification, not as the executable source of the Task B comparison calculations. The current runner requires these files to complete. | Document as historical verification dependencies; do not copy. Provide separately only if rights and institutional approval permit. Otherwise mark Task B verification as unavailable in any public execution instructions. |
| **Task B outputs (Tasks C and D)** | Task C uses Task B verification and run-summary values in graph preparation and baseline verification; Task D uses Task B run summaries and its finalizer checks Task B output hashes. | Required inter-task artifacts for the staged workflow. Do not include restricted song-level or derived files in a public package; if not lawfully shareable, document the prerequisite and limitations. |
| **Task C outputs (Tasks D and E)** | Graph, IDs, and partition arrays are consumed by the central downstream analyses; validation records are used by Task D finalization. | Required computational inputs, but restricted arrays/IDs are not staged. Provide only with explicit rights approval; otherwise mark execution dependent on unavailable inputs. |
| **Task A/A2 manifest and source artifacts (Task D finalizer)** | Used to verify prior characterization evidence integrity; not needed by the Task D quantitative runner itself. | Validation-only. Do not copy; document that the optional finalizer check cannot run without authorized Task A/A2 evidence. |
| **WP4 Rev1 code/config/summary and saved metrics (Task E)** | Metrics and artist table feed descriptive Rev1-vs-Rev2 comparisons; WP4 script/config/summary are hash/provenance inputs. They are required by the staged Task E script as currently written but do not implement the central Rev2 calculations. | Document as historical comparison/validation inputs; do not copy. Supply separately only after rights approval. If unavailable, the script cannot complete as written and should be described as such; no substitute is created here. |

The package contains no vendored third-party library source. It is released under MIT at the copyright claimant's direction. Dependencies retain their own licences; this package's MIT licence does not relicense them.

## Reproducibility coverage

| Evidence level | Status |
|---|---|
| Script source exists and is staged | Yes, for the eight listed B–E scripts. |
| Parameters and many inputs documented | Yes, in the task provenance and validation records summarized here; dependency and access requirements remain incomplete. |
| Execution is plausible with compatible inputs | Yes, conditionally; required inputs and historical dependencies are not part of this package. |
| Current staged code independently rerun | No. This preparation performed static checks only. |
| Complete historical corpus preparation and embedding generation reproduced | No; not established. |
| Published results independently reproduced from this package | No. |

## Release scope

The GitHub release is versioned as `v1.0.0`. No Zenodo deposit or DOI is provided. The package contains no study dataset, lyric excerpts, embeddings, graph or partition arrays, song-level records or identifiers, or generated outputs. Schema names and code that reads private inputs remain necessary parts of the analytical scripts. Public artist names used in the study's comparisons are retained.
