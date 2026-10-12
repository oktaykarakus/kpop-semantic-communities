"""Validate Task D outputs and write the bounded scientific report."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KPOP_PROJECT_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()
OUT = Path(os.environ.get("KPOP_TASK_D_OUTPUT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def verify_prior_packages() -> dict[str, object]:
    checks: dict[str, object] = {}
    task_a_manifest = json.loads(
        (ROOT / "revision/Rev2_TaskA_community_characterisation/input_and_code_manifest.json")
        .read_text(encoding="utf-8")
    )
    marker = "K-Pop Project/"
    task_a_hash_checks = {}
    for item in task_a_manifest:
        old_path = Path(item["path"])
        old_text = old_path.as_posix()
        if marker not in old_text:
            raise ValueError(f"Unexpected Task A manifest path: {old_text}")
        relative = old_text.split(marker, maxsplit=1)[1]
        path = ROOT / relative
        actual = sha256(path)
        if actual != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Historical Task A input or executed code changed: {relative}")
        task_a_hash_checks[relative] = actual
    checks["task_a_manifest_entries_verified"] = task_a_hash_checks

    task_a2_hashes = {
        "community_profiles.csv": "b992db7492a971cc5345d9c45fe40803bac2fc3757b387e0f771fc87a56003fc",
        "community_representative_songs.csv": "b7740f042e3eb26b6c89f08a41addc20981eb86a0262904ae4fcc859c78e5bac",
        "community_terms.csv": "7081be03f919be7de6b77cbb792b10d780091d338396d863d8d73ee67d37c7e1",
        "song_language_script.csv": "2444f8f82057203dbad1dc329c355021d957dca6b1412c72b5fceda40145ad20",
        "community_language_script_summary.csv": "4e2321594d20e4fa83484e4823c5f6ab81a54c402b00c3fd138f85913e799b5f",
        "TaskA_summary.md": "c21fc64f63cb0a591d37aee0ef3aed74a3abf96a19aff31b6318c4331af3bd1b",
        "provenance.md": "d7814eea8f755c8de0a406176561a3dee368ec07bb1ef58b66ac73eacccf0545",
    }
    task_a_dir = ROOT / "revision/Rev2_TaskA_community_characterisation"
    task_a2_checks = {}
    for name, expected in task_a2_hashes.items():
        actual = sha256(task_a_dir / name)
        if actual != expected:
            raise ValueError(f"Task A source for the historical A2 interpretation changed: {name}")
        task_a2_checks[name] = actual
    checks["task_a2_source_hashes_verified"] = task_a2_checks

    task_b_dir = Path(
        os.environ.get(
            "KPOP_TASK_B_OUTPUT_DIR", ROOT / "revision/Rev2_TaskB_representation_comparison"
        )
    ).expanduser().resolve()
    task_b_validation = json.loads(
        (task_b_dir / "output_validation.json").read_text(encoding="utf-8")
    )
    task_b_checks = {}
    for name, details in task_b_validation.items():
        if not isinstance(details, dict) or "sha256" not in details:
            continue
        actual = sha256(task_b_dir / name)
        if actual != details["sha256"]:
            raise ValueError(f"Task B artifact changed: {name}")
        task_b_checks[name] = actual
    if task_b_checks.get("louvain_repeated_runs.csv") is None:
        raise ValueError("Task B run summary integrity hash was not verified.")
    checks["task_b_output_hashes_verified"] = task_b_checks

    task_c_dir = Path(
        os.environ.get(
            "KPOP_TASK_C_OUTPUT_DIR", ROOT / "revision/Rev2_TaskC_alternative_communities"
        )
    ).expanduser().resolve()
    task_c_validation = json.loads(
        (task_c_dir / "output_validation.json").read_text(encoding="utf-8")
    )
    task_c_checks = {}
    for name, expected in task_c_validation["output_sha256"].items():
        actual = sha256(task_c_dir / name)
        if actual != expected:
            raise ValueError(f"Task C result changed: {name}")
        task_c_checks[name] = actual
    graph_validation = json.loads(
        (task_c_dir / "graph_verification.json").read_text(encoding="utf-8")
    )
    graph_sha = sha256(task_c_dir / "graph.npz")
    if graph_sha != graph_validation["graph_sha256"]:
        raise ValueError("Task C whole-song graph changed.")
    checks["task_c_output_hashes_verified"] = task_c_checks
    checks["task_c_graph_sha256_verified"] = graph_sha
    return checks


def main() -> None:
    for name in ("TaskD_summary.md", "provenance.md", "output_validation.json"):
        if (OUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite {name}.")

    partition = pd.read_csv(OUT / "final_partition.csv", dtype={"song_uid": str})
    selection = pd.read_csv(OUT / "partition_selection.csv")
    profiles = pd.read_csv(OUT / "community_profiles.csv")
    representatives = pd.read_csv(
        OUT / "community_representative_songs.csv", dtype={"song_uid": str}
    )
    terms = pd.read_csv(OUT / "community_terms.csv")
    scripts = pd.read_csv(OUT / "community_script_summary.csv")
    association = pd.read_csv(OUT / "script_association.csv").iloc[0]
    interpretation = pd.read_csv(OUT / "community_interpretation_table.csv")
    evidence = pd.read_csv(
        OUT / "selected_interpretation_evidence.csv", dtype={"song_uid": str}
    )
    metadata = json.loads((OUT / "analysis_metadata.json").read_text(encoding="utf-8"))

    corpus = pd.read_parquet(ROOT / "data/processed_revision1/lyrics.parquet")
    canonical_ids = set(corpus["song_uid"].astype(str))
    if (
        len(partition) != 7891
        or partition["song_uid"].isna().any()
        or partition["song_uid"].nunique() != 7891
        or partition["community"].isna().any()
        or set(partition["song_uid"]) != canonical_ids
    ):
        raise ValueError("Final partition does not assign every canonical song exactly once.")
    task_c = ROOT / "revision/Rev2_TaskC_alternative_communities"
    saved_ids = np.load(task_c / "song_ids.npy", allow_pickle=False).astype(str)
    saved_labels = np.load(task_c / "baseline_labels.npy", allow_pickle=False)
    selected_seed = int(metadata["selected_seed"])
    if not np.array_equal(partition["song_uid"].to_numpy(), saved_ids):
        raise ValueError("Final partition IDs/order do not match Task C whole-song graph IDs.")
    if not np.array_equal(partition["community"].to_numpy(), saved_labels[selected_seed]):
        raise ValueError("Final labels are not the selected saved whole-song Louvain partition.")
    if int(metadata["selected_community_count"]) != profiles["community"].nunique():
        raise ValueError("Profile count differs from selected partition community count.")
    size_by_community = partition["community"].value_counts().sort_index()
    profile_sizes = profiles.set_index("community")["n_songs"].sort_index()
    if not size_by_community.equals(profile_sizes.astype(int)):
        raise ValueError("Community profile sizes differ from final partition.")
    if int(profile_sizes.sum()) != 7891:
        raise ValueError("Community sizes do not sum to the canonical corpus size.")
    if not set(representatives["song_uid"]).issubset(canonical_ids):
        raise ValueError("Representative-song IDs are not all in the canonical corpus.")
    if not set(evidence["song_uid"]).issubset(canonical_ids):
        raise ValueError("Interpretation-evidence IDs are not all in the canonical corpus.")
    assignments = partition.set_index("song_uid")["community"]
    if not representatives.apply(
        lambda row: assignments.loc[row["song_uid"]] == row["community"], axis=1
    ).all():
        raise ValueError("Representative song community labels are inconsistent.")
    if not evidence.apply(
        lambda row: assignments.loc[row["song_uid"]] == row["community"], axis=1
    ).all():
        raise ValueError("Interpretation evidence community labels are inconsistent.")
    communities = sorted(partition["community"].unique().tolist())
    if (
        profiles["community"].tolist() != communities
        or scripts["community"].tolist() != communities
        or interpretation["community"].tolist() != communities
        or terms.groupby("community").size().reindex(communities).ne(20).any()
    ):
        raise ValueError("Community-level outputs do not align or contain 20 terms per group.")
    if set(interpretation["interpretation_category"]) != {
        "Theme-driven",
        "Mixed content and language/form",
        "Primarily language/form",
        "Heterogeneous/unclear",
    }:
        raise ValueError("Interpretation categories are missing or inconsistent.")
    if interpretation["community"].duplicated().any():
        raise ValueError("Interpretation table must contain exactly one row per community.")
    if int(scripts["n_songs"].sum()) != 7891:
        raise ValueError("Community-script totals do not sum to 7,891.")
    stratum_columns = [f"{stratum}_n" for stratum in (
        "hangul_dominant",
        "mixed_hangul_latin",
        "latin_dominant",
        "other_script",
        "no_letters",
    )]
    if int(scripts[stratum_columns].to_numpy().sum()) != 7891:
        raise ValueError("Script contingency cells do not sum to 7,891.")
    if int(association["n_songs"]) != 7891:
        raise ValueError("Script association used an unexpected corpus size.")
    if int(evidence.groupby("community").size().min()) < 10:
        raise ValueError("At least ten interpretation excerpts per community are required.")

    sizes = size_by_community.astype(int).sort_values(ascending=False).tolist()
    category_counts = (
        interpretation["interpretation_category"].value_counts().to_dict()
    )
    stratum_totals = metadata["script_association"][
        "hangul_dominant_mixed_latin_dominant_other_and_no_letters_counts"
    ]
    script_counts = {
        stratum: int(stratum_totals[stratum])
        for stratum in (
            "hangul_dominant",
            "mixed_hangul_latin",
            "latin_dominant",
            "other_script",
            "no_letters",
        )
    }
    integrity = verify_prior_packages()
    versions = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "networkx": importlib.metadata.version("networkx"),
        "pyarrow": importlib.metadata.version("pyarrow"),
    }
    input_hashes = metadata["input_sha256"]
    for relative, digest in input_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Task D input changed since analysis: {relative}")

    summary = f"""# Rev2 Task D — final whole-song community characterization

## Final operational partition

The representative partition is whole-song-truncated Louvain seed **{selected_seed}**, selected from the 20 saved Task B whole-song runs as the partition with the highest mean NMI to the other 19. Its mean NMI is **{metadata['selected_mean_nmi']:.3f}**, mean ARI is **{metadata['selected_mean_ari']:.3f}**, and Task B reported modularity is **{metadata['selected_task_b_reported_modularity']:.6f}**. It has **{len(communities)} communities** with sizes, descending: {', '.join(f'{size:,}' for size in sizes)}. Seed 19 wins by the NMI criterion; the modularity/seed tie-breakers were not needed.

This selection is a representative operational partition, not a finding that the number or boundaries of communities are intrinsic. Task C found broad cross-method overlap but no robust PyGenStability plateau near the Louvain count range. The fixed count should therefore be reported as the outcome of the stated graph, representation, resolution, and representative-seed rule.

## Script association

Community membership is clearly associated with character-script composition: Pearson chi-square **{association['chi_square']:.2f}** (df={int(association['degrees_of_freedom'])}), Cramér's V **{association['cramers_v']:.3f}** (bias-corrected **{association['bias_corrected_cramers_v']:.3f}**), with song-label permutation p={association['permutation_p_value']:.3f} from {int(association['permutations'])} permutations. Corpus strata were {script_counts['hangul_dominant']:,} Hangul-dominant, {script_counts['mixed_hangul_latin']:,} mixed Hangul–Latin, {script_counts['latin_dominant']:,} Latin-dominant, {script_counts['other_script']:,} other-script, and {script_counts['no_letters']:,} no-letter songs. Script composition varies substantially among groups: C0 is entirely Hangul-dominant, while C15 is mostly mixed-script and has a mean Latin-character share above its mean Hangul share.

This association is meaningful but smaller in magnitude than Task A's Cramér's V of 0.484 for the old line-mean partition; the graph representation and partition changed, so that contrast is contextual rather than a controlled causal comparison. Twenty-three expected contingency cells are below five, so the asymptotic chi-square p-value is not relied on; the permutation p-value is the reported reference. Even that test treats songs as exchangeable and does not model artist/version dependence or graph dependence. Script is a character-composition proxy, not validated language identification, and association is not evidence of causation.

## Bounded content interpretation

The reviewed representatives and discriminative terms show recurring but overlapping lyrical tendencies. They include seasonal/weather imagery around memory (C2), night-sky imagery and longing (C3), self-reflection and resilience (C4), breakup grief and place-linked memory (C8), remembering after separation (C10), relationship hurt and pleas (C11), love declarations and difficulty letting go (C12), and focus on an unavailable “her” (C19). Music as a medium for memory or address appears in C1. These are descriptions of selected evidence, not prevalence estimates or validated topic labels.

The interpretation table classifies **{category_counts.get('Theme-driven', 0)}** communities as theme-driven, **{category_counts.get('Mixed content and language/form', 0)}** as mixed content and language/form, **{category_counts.get('Primarily language/form', 0)}** as primarily language/form, and **{category_counts.get('Heterogeneous/unclear', 0)}** as heterogeneous/unclear. C15 is the clearest primarily language/form case: dance/performance address, repeated hooks and vocables, alongside a high mixed-script share. C1, C5, C7, C9, C17, and C18 combine identifiable relationship or attraction content with conspicuous music-reference, hook, English-address, or script-mixing cues; they are not treated as form-only.

Overall, the evidence is **consistent with meaningful but non-exclusive thematic organization co-occurring with language/script/form patterning**. It does not support 21 distinct, pure semantic topics: broad love, separation, longing, and direct address recur across multiple groups, and individual communities contain counterexamples.

## Defensible claims and claims to avoid

The manuscript can report the reproducible 21-group Louvain partition as a practical summary of structure in the chosen whole-song k=30 graph, report its measured association with script proxies, and describe a small number of recurring lyrical patterns in explicitly bounded representative evidence. It can say content and script/form cues co-occur and that at least one group is most cautiously described through performance and hook form.

Avoid claiming that community count is intrinsic or robust across methods; that every group is a distinct semantic topic; that script association proves language causes clustering; that the script proxy is language identification; that selected excerpts characterize every member; or that these qualitative labels have human validation. Whole-song truncation may omit later lyric content. The partition uses the same embedding space that is profiled, so the descriptive evidence is not independent semantic validation.
"""
    write_exclusive(OUT / "TaskD_summary.md", summary)

    provenance = f"""# Rev2 Task D provenance

Run environment: Python {versions['python']} at `{sys.executable}`; NumPy {versions['numpy']}, pandas {versions['pandas']}, SciPy {versions['scipy']}, scikit-learn {versions['scikit-learn']}, NetworkX {versions['networkx']}, PyArrow {versions['pyarrow']}. No dependency installation or external web/API access was performed.

All new files are confined to `revision/Rev2_TaskD_final_characterisation/`. This task did not write to the manuscript, Rev1, Tasks A/A2/B/C, or the canonical data directories. `company_semantics/` was not accessed. Task A2 outputs and methodological notes were used only as methodological references; old line-mean labels were not used to assign any final community. Task C's saved whole-song graph, IDs, and 20 baseline label arrays were reused; neither Louvain nor PyGenStability was rerun.

## Partition selection

Input partitions: `revision/Rev2_TaskC_alternative_communities/baseline_labels.npy`, aligned to `song_ids.npy`, exactly matching the whole-song-truncated embedding-ID order. For each seed 0–19, pairwise NMI and ARI were computed against all other seeds and averaged over the 19 distinct comparisons. Select maximum mean NMI; exact ties would be broken by higher Task B reported modularity, then lower seed. Selected seed {selected_seed}; mean NMI {metadata['selected_mean_nmi']:.12f}; mean ARI {metadata['selected_mean_ari']:.12f}; Task B reported modularity {metadata['selected_task_b_reported_modularity']:.12f}. Task B's reported modularity is standard weighted modularity for the selected partition (gamma=1 evaluation), not a resolution-2 modularity objective. Its counts and modularity were independently checked against the saved Task B rows using the reused Task C graph.

The selected partition has {len(communities)} groups; sorted sizes: {', '.join(map(str, sizes))}. Exact per-song labels are preserved in `final_partition.csv`, with one row per saved song ID and no line-mean join.

## Profiles and interpretation

Profiles reuse Task A's text cleaning (remove square-bracket segments; collapse whitespace), TF–IDF configuration (lowercase, English stop-word list, min_df=3, max_df=0.90, maximum 30,000 features, unigrams/bigrams, `(?u)\\b\\w\\w+\\b` token pattern), and within-community minus outside-community mean TF–IDF ranking. Twenty terms are saved per community.

Existing 384-dimensional whole-song vectors are L2-normalized only for centroid ranking; no embeddings are regenerated. Representatives are the union of the ten highest centroid-cosine and ten highest within-community weighted PageRank songs (alpha=.85, tolerance=1e-6, max_iter=100), with ties resolved by song ID. Profiles report song and artist counts, top-artist concentration/HHI, centroid cosine summaries, terms, ranked songs, and leading excerpts. Artist spellings are not entity-resolved.

For interpretation evidence, centroid and PageRank ranks were sorted separately; the two lists were alternated starting with centroid, duplicate song IDs skipped, and the first 12 unique IDs retained (or all available for smaller communities). `selected_interpretation_evidence.csv` saves these selected excerpts and ranks. The descriptive synthesis reviewed ranks 1, 4, 8, and 12 per community, with additional rank-6 spot checks where useful, and all 20 discriminative terms and profile/script summaries. No formal motif coding, external model/API, or human validation was performed.

## Script proxy and association

Count Unicode alphabetic characters by Unicode name: HANGUL, LATIN, and other letters. Spaces, punctuation, and digits are excluded; repeated lyrics/choruses count as present. Strata use Task A thresholds: Hangul share >=0.80; else Latin share >=0.80; else combined Hangul+Latin share >=0.80 is mixed; otherwise other-script; a zero-letter record would be no_letters. This is a character-script proxy, not validated language identification.

The community-by-stratum Pearson chi-square is uncorrected. Cramér's V and bias-corrected V are reported; p-values include 999 global song-label permutations with NumPy default_rng seed 42 and the plus-one estimator. Continuous Hangul and Latin share eta-squared values and analogous permutation references are recorded in `script_association.csv`. These tests are exploratory, with no adjustment for multiple comparisons, artist/version clustering, or graph dependence; effect sizes receive priority.

## Verification and prior-artifact integrity

The finalizer checked exact 7,891-row unique-ID coverage and equality to the selected saved label array; complete nonmissing assignments; community sizes summing to 7,891; representative and interpretation-evidence IDs and labels against the canonical corpus and selected partition; exactly 20 terms per community; full community-script counts summing to 7,891; and category coverage. The selected whole-song input hashes are recorded in `analysis_metadata.json`.

Prior-package integrity checks verified every data/code entry in the Task A manifest, all seven Task A source checksums recorded in Task A2 provenance, all hashed Task B outputs in its validation file, and all Task C outputs and graph checksums in its validation records. These include the historical Rev1 graph and line-mean partition hashes. No Rev1 manuscript files were read or edited; writes were restricted to this new Task D directory.

## Input SHA-256

{chr(10).join(f'- `{relative}`: `{digest}`' for relative, digest in input_hashes.items())}
"""
    write_exclusive(OUT / "provenance.md", provenance)

    required_outputs = [
        "final_partition.csv",
        "partition_selection.csv",
        "community_profiles.csv",
        "community_representative_songs.csv",
        "community_terms.csv",
        "community_script_summary.csv",
        "script_association.csv",
        "community_interpretation_table.csv",
        "community_interpretation_notes.md",
        "TaskD_summary.md",
        "provenance.md",
    ]
    output_hashes = {name: sha256(OUT / name) for name in required_outputs}
    validation = {
        "status": "passed",
        "canonical_songs": 7891,
        "unique_partition_song_ids": int(partition["song_uid"].nunique()),
        "missing_community_assignments": int(partition["community"].isna().sum()),
        "community_count": len(communities),
        "community_sizes_sum": int(profile_sizes.sum()),
        "script_contingency_total": int(scripts[stratum_columns].to_numpy().sum()),
        "representative_song_ids_in_canonical_corpus": bool(
            set(representatives["song_uid"]).issubset(canonical_ids)
        ),
        "interpretation_evidence_rows": len(evidence),
        "one_interpretation_row_per_community": len(interpretation) == len(communities),
        "selected_labels_match_saved_task_c_whole_song_louvain": True,
        "old_line_mean_partition_used": False,
        "task_a_a2_b_c_prior_artifact_checks_passed": True,
        "rev1_graph_and_partition_manifest_hashes_passed": True,
        "manuscript_files_modified": False,
        "company_semantics_accessed": False,
        "category_counts": category_counts,
        "prior_integrity_checks": integrity,
        "output_sha256": output_hashes,
    }
    write_exclusive(
        OUT / "output_validation.json",
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "status": validation["status"],
                "selected_seed": selected_seed,
                "community_count": len(communities),
                "community_sizes": sizes,
                "script_cramers_v": float(association["cramers_v"]),
                "category_counts": category_counts,
                "verified_prior_packages": [
                    "Rev1 manifest files",
                    "Task A",
                    "Task A2",
                    "Task B",
                    "Task C",
                ],
                "required_output_files": required_outputs + ["output_validation.json"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
