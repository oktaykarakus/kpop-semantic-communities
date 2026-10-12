from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(os.environ.get("KPOP_PROJECT_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()
OUT = Path(os.environ.get("KPOP_TASK_E_OUTPUT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
DATA = ROOT / "data/processed_revision1"
WP4 = ROOT / "revision/WP4_revision1_boundary_validation"
TASK_C = Path(os.environ.get("KPOP_TASK_C_OUTPUT_DIR", ROOT / "revision/Rev2_TaskC_alternative_communities")).expanduser().resolve()
TASK_D = Path(os.environ.get("KPOP_TASK_D_OUTPUT_DIR", ROOT / "revision/Rev2_TaskD_final_characterisation")).expanduser().resolve()

GRAPH_PATH = TASK_C / "graph.npz"
GRAPH_IDS_PATH = TASK_C / "song_ids.npy"
EMBEDDINGS_PATH = DATA / "lyrics_embeddings_whole_song_truncated.npy"
EMBEDDING_IDS_PATH = DATA / "lyrics_embedding_ids_whole_song_truncated.npy"
CORPUS_PATH = DATA / "lyrics.parquet"
PARTITION_PATH = TASK_D / "final_partition.csv"
INTERPRETATION_PATH = TASK_D / "community_interpretation_table.csv"
EVIDENCE_PATH = TASK_D / "selected_interpretation_evidence.csv"
WP4_SCRIPT_PATH = WP4 / "run_wp4_revision1.py"
WP4_CONFIG_PATH = WP4 / "config.py"
WP4_SUMMARY_PATH = WP4 / "outputs/WP4_summary.md"
WP4_METRICS_PATH = WP4 / "outputs/raw/boundary_metrics_all_songs.parquet"
WP4_ARTISTS_PATH = WP4 / "outputs/tables/artist_boundary_summary_all.csv"

MIN_SONGS_PER_ARTIST = 5
N_BOOTSTRAPS = 1000
RANDOM_SEED = 42
BOUNDARY_QUANTILES = (0.80, 0.85, 0.90, 0.95)
EXPECTED_NODES = 7891
EXPECTED_COMMUNITIES = 21
FOCAL_ARTISTS = {"BTS": "방탄소년단", "BLACKPINK": "BLACKPINK"}
STAGING_ROOT = Path(__file__).resolve().parents[2]


def load_example_lyric_summaries() -> tuple[dict[str, str], Path | None]:
    annotation_file = os.environ.get("KPOP_TASK_E_ANNOTATIONS")
    if annotation_file is None:
        return {}, None
    path = Path(annotation_file).expanduser().resolve()
    if path == STAGING_ROOT or STAGING_ROOT in path.parents:
        raise ValueError("Task E annotation mapping must be stored outside the release package.")
    with path.open(encoding="utf-8") as stream:
        annotations = json.load(stream)
    if not isinstance(annotations, dict) or len(annotations) > 5 or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in annotations.items()
    ):
        raise ValueError(
            "Task E example annotations must be a JSON object of at most five string keys and values."
        )
    return annotations, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate():
    graph = sp.load_npz(GRAPH_PATH).tocsr()
    graph_ids = np.load(GRAPH_IDS_PATH, allow_pickle=False).astype(str)
    embedding_ids = np.load(EMBEDDING_IDS_PATH, allow_pickle=True).astype(str)
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    partition = pd.read_csv(PARTITION_PATH)
    partition["song_uid"] = partition["song_uid"].astype(str)
    corpus = pd.read_parquet(CORPUS_PATH)
    corpus["song_uid"] = corpus["song_uid"].astype(str)
    interpretation = pd.read_csv(INTERPRETATION_PATH)
    evidence = pd.read_csv(EVIDENCE_PATH)
    old_metrics = pd.read_parquet(WP4_METRICS_PATH)
    old_artists = pd.read_csv(WP4_ARTISTS_PATH)

    labels = partition.set_index("song_uid")["community"].reindex(graph_ids)
    if labels.isna().any():
        raise ValueError("Task D partition is missing graph-node assignments.")
    labels_array = labels.astype(int).to_numpy()

    if graph.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise ValueError(f"Unexpected graph shape: {graph.shape}")
    if len(graph_ids) != EXPECTED_NODES or len(set(graph_ids)) != EXPECTED_NODES:
        raise ValueError("Task C graph node IDs are not exactly 7,891 unique IDs.")
    if not np.array_equal(graph_ids, embedding_ids):
        raise ValueError("Task C graph node order differs from whole-song embedding IDs.")
    if not np.array_equal(graph_ids, partition["song_uid"].to_numpy()):
        raise ValueError("Task D partition row order differs from Task C graph node order.")
    if partition["song_uid"].nunique() != EXPECTED_NODES:
        raise ValueError("Task D partition does not contain exactly 7,891 unique IDs.")
    if set(np.unique(labels_array)) != set(range(EXPECTED_COMMUNITIES)):
        raise ValueError("Task D partition is not the expected 21-community partition.")
    if embeddings.shape != (EXPECTED_NODES, 384) or not np.isfinite(embeddings).all():
        raise ValueError(f"Unexpected or invalid whole-song embeddings: {embeddings.shape}")
    if len(corpus) != EXPECTED_NODES or corpus["song_uid"].nunique() != EXPECTED_NODES:
        raise ValueError("Canonical lyrics corpus is not exactly 7,891 unique songs.")
    if set(graph_ids) != set(corpus["song_uid"]):
        raise ValueError("Graph nodes and canonical corpus IDs differ.")
    if not (graph - graph.T).nnz == 0:
        raise ValueError("Frozen graph adjacency is not symmetric.")
    if graph.data.size == 0 or not np.isfinite(graph.data).all():
        raise ValueError("Frozen graph has missing or non-finite edge weights.")
    if (graph.data < 0).any():
        raise ValueError("Frozen graph contains negative cosine edge weights.")

    corpus = (
        pd.DataFrame({"song_uid": graph_ids, "community": labels_array})
        .merge(corpus, on="song_uid", how="left", validate="one_to_one")
    )
    if corpus["artist"].isna().any():
        raise ValueError("Canonical corpus contains missing artist values.")
    return graph, graph_ids, embeddings, labels_array, corpus, interpretation, evidence, old_metrics, old_artists


def normalized_entropy(strengths: np.ndarray) -> float:
    positive = strengths[strengths > 0]
    if len(positive) <= 1:
        return 0.0
    proportions = positive / positive.sum()
    entropy = -np.sum(proportions * np.log(proportions))
    return float(entropy / np.log(len(strengths)))


def compute_metrics(graph, ids, embeddings, labels):
    communities = np.unique(labels)
    community_to_index = {int(c): i for i, c in enumerate(communities)}
    centroids = np.vstack(
        [np.asarray(embeddings[labels == c]).mean(axis=0) for c in communities]
    )
    norms = np.linalg.norm(centroids, axis=1)
    centroids = centroids / np.maximum(norms[:, None], 1e-12)
    centroid_sim = np.asarray(embeddings @ centroids.T)
    if centroid_sim.shape != (len(ids), len(communities)):
        raise ValueError("Centroid similarity dimensions do not match the corpus.")
    top_two = np.sort(centroid_sim, axis=1)[:, ::-1][:, :2]
    centroid_margin = top_two[:, 0] - top_two[:, 1]
    centroid_novelty = 1.0 - top_two[:, 0]

    rows = []
    for i, uid in enumerate(ids):
        start, stop = graph.indptr[i], graph.indptr[i + 1]
        neighbours = graph.indices[start:stop]
        weights = graph.data[start:stop].astype(float, copy=False)
        strength_by_community = np.bincount(
            [community_to_index[int(labels[j])] for j in neighbours],
            weights=weights,
            minlength=len(communities),
        )
        total = float(strength_by_community.sum())
        own_index = community_to_index[int(labels[i])]
        own_strength = float(strength_by_community[own_index])
        external_strength = total - own_strength
        reached = strength_by_community > 0
        external_reached = reached.copy()
        external_reached[own_index] = False
        proportions = strength_by_community / total if total > 0 else strength_by_community
        participation = (
            float(1.0 - np.square(proportions).sum()) if total > 0 else 0.0
        )
        degree = int(len(neighbours) + (1 if i in neighbours else 0))
        rows.append(
            {
                "song_uid": uid,
                "community": int(labels[i]),
                "degree": degree,
                "weighted_degree": total,
                "internal_strength": own_strength,
                "external_strength": external_strength,
                "external_strength_fraction": external_strength / total if total > 0 else 0.0,
                "participation_coefficient": participation,
                "community_entropy": normalized_entropy(strength_by_community),
                "communities_reached": int(reached.sum()),
                "external_communities_reached": int(external_reached.sum()),
                "mean_neighbour_cosine": float(weights.mean()) if len(weights) else 0.0,
                "local_novelty": 1.0 - float(weights.mean()) if len(weights) else 1.0,
                "max_centroid_cosine": float(top_two[i, 0]),
                "second_centroid_cosine": float(top_two[i, 1]),
                "centroid_margin": float(centroid_margin[i]),
                "centroid_novelty": float(centroid_novelty[i]),
                "bridge_affinity_score": float(
                    top_two[i, 1] * (1.0 - centroid_margin[i])
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = np.empty(N_BOOTSTRAPS, dtype=float)
    for i in range(N_BOOTSTRAPS):
        draws[i] = rng.choice(values, size=len(values), replace=True).mean()
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def artist_metrics(metrics: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    merged = metrics
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for artist, group in merged.groupby("artist", sort=True):
        if len(group) < MIN_SONGS_PER_ARTIST:
            continue
        values = group["participation_coefficient"].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, rng)
        rows.append(
            {
                "artist": artist,
                "n_songs": int(len(group)),
                "mean_boundary_score": float(values.mean()),
                "median_boundary_score": float(np.median(values)),
                "std_boundary_score": float(values.std(ddof=1)),
                "se_boundary_score": float(values.std(ddof=1) / np.sqrt(len(values))),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "mean_communities_reached": float(
                    group["external_communities_reached"].mean()
                ),
                "n_communities_present": int(group["community"].nunique()),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["mean_boundary_score", "artist"], ascending=[False, True]
    ).reset_index(drop=True)
    result.insert(0, "rank_mean_boundary_score", np.arange(1, len(result) + 1))
    return result


def q90_results(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    score = metrics["participation_coefficient"].to_numpy(dtype=float)
    threshold = float(np.quantile(score, 0.90))
    boundary = score >= threshold
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for variable in (
        "external_strength_fraction",
        "community_entropy",
        "external_communities_reached",
        "centroid_novelty",
        "local_novelty",
        "degree",
        "weighted_degree",
    ):
        x = metrics.loc[boundary, variable].to_numpy(dtype=float)
        y = metrics.loc[~boundary, variable].to_numpy(dtype=float)
        test = mannwhitneyu(x, y, alternative="two-sided")
        draws = np.empty(N_BOOTSTRAPS, dtype=float)
        for i in range(N_BOOTSTRAPS):
            xb = rng.choice(x, size=len(x), replace=True)
            yb = rng.choice(y, size=len(y), replace=True)
            draws[i] = xb.mean() - yb.mean()
        rows.append(
            {
                "variable": variable,
                "boundary_mean": float(x.mean()),
                "non_boundary_mean": float(y.mean()),
                "mean_difference": float(x.mean() - y.mean()),
                "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                "mann_whitney_u": float(test.statistic),
                "mann_whitney_p": float(test.pvalue),
                "cliffs_delta": float(
                    2.0 * test.statistic / (len(x) * len(y)) - 1.0
                ),
            }
        )
    result = {
        "q90_threshold": threshold,
        "n_boundary": int(boundary.sum()),
        "n_non_boundary": int((~boundary).sum()),
    }
    return pd.DataFrame(rows), result


def novelty_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in (
        "centroid_novelty",
        "local_novelty",
        "second_centroid_cosine",
        "centroid_margin",
        "external_communities_reached",
        "bridge_affinity_score",
    ):
        result = spearmanr(
            metrics["participation_coefficient"],
            metrics[variable],
        )
        rows.append(
            {
                "boundary_score": "participation_coefficient",
                "variable": variable,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def norm_title(title: str) -> str:
    title = re.sub(r"\([^)]*\)", " ", str(title).casefold())
    return "".join(ch for ch in unicodedata.normalize("NFKC", title) if ch.isalnum())


def select_examples(
    metrics: pd.DataFrame,
    corpus: pd.DataFrame,
    interpretation: pd.DataFrame,
    evidence: pd.DataFrame,
    graph: sp.csr_matrix,
    ids: np.ndarray,
    labels: np.ndarray,
    example_lyric_summaries: dict[str, str],
) -> pd.DataFrame:
    labels_by_id = dict(zip(ids, labels))
    descriptions = interpretation.set_index("community")[
        ["descriptive_label", "interpretation_category", "confidence"]
    ].to_dict("index")
    evidence_small = evidence[["song_uid", "lyric_snippet"]].drop_duplicates("song_uid")
    candidates = (
        evidence[["song_uid", "community", "title", "artist"]]
        .merge(
            metrics.drop(columns=["artist", "title", "date"], errors="ignore"),
            on=["song_uid", "community"],
            how="inner",
            validate="one_to_one",
        )
        .merge(evidence_small, on="song_uid", how="left", validate="one_to_one")
        .sort_values(["participation_coefficient", "song_uid"], ascending=[False, True])
    )
    selected = []
    seen_communities: set[int] = set()
    seen_artists: set[str] = set()
    seen_titles: set[str] = set()
    for row in candidates.itertuples(index=False):
        normalized = norm_title(row.title)
        if (
            row.community in seen_communities
            or row.artist in seen_artists
            or normalized in seen_titles
            or row.external_strength <= 0
        ):
            continue
        i = int(np.where(ids == row.song_uid)[0][0])
        start, stop = graph.indptr[i], graph.indptr[i + 1]
        neighbours = graph.indices[start:stop]
        weights = graph.data[start:stop]
        cross: dict[int, float] = {}
        for j, weight in zip(neighbours, weights):
            c = int(labels[j])
            if c != int(row.community):
                cross[c] = cross.get(c, 0.0) + float(weight)
        total_external = sum(cross.values())
        top_neighbours = sorted(cross.items(), key=lambda item: (-item[1], item[0]))[:3]
        neighbour_info = [
            {
                "community": c,
                "cross_edge_strength": round(s, 6),
                "share_of_external_strength": round(s / total_external, 4),
                "interpretation": descriptions[c]["descriptive_label"],
            }
            for c, s in top_neighbours
        ]
        own = descriptions[int(row.community)]
        selected.append(
            {
                "song_id": row.song_uid,
                "song": row.title,
                "artist": row.artist,
                "community": int(row.community),
                "community_interpretation": own["descriptive_label"],
                "participation_coefficient": float(row.participation_coefficient),
                "external_strength_fraction": float(row.external_strength_fraction),
                "top_neighbouring_communities": json.dumps(
                    neighbour_info, ensure_ascii=False
                ),
                "observed_graph_evidence": (
                    "Weighted k=30 links distribute across the listed other communities; "
                    f"external strength fraction={row.external_strength_fraction:.3f}."
                ),
                "lyric_excerpt_observation": example_lyric_summaries.get(
                    row.song_uid,
                    "No song-specific lyric observation was added.",
                ),
                "bounded_interpretation": (
                    "The excerpt and neighboring community descriptions share broad "
                    "relationship/separation motifs. This does not independently establish "
                    "lyrical hybridity or a song-level semantic blend."
                ),
                "interpretation_confidence": (
                    "Moderate for the short excerpt; low for generalizing to the full song "
                    "or its community"
                ),
                "task_d_evidence_available": bool(
                    isinstance(row.lyric_snippet, str) and row.lyric_snippet
                ),
            }
        )
        seen_communities.add(int(row.community))
        seen_artists.add(str(row.artist))
        seen_titles.add(normalized)
        if len(selected) == 5:
            break
    return pd.DataFrame(selected)


def compare_focal_artists(
    current_artists: pd.DataFrame,
    old_artists: pd.DataFrame,
    current_corpus: pd.DataFrame,
    old_metrics: pd.DataFrame,
) -> pd.DataFrame:
    current_by_name = current_artists.set_index("artist")
    old_by_name = old_artists.set_index("artist")
    old_song_artist = old_metrics.merge(
        current_corpus[["song_uid", "artist"]],
        on="song_uid",
        how="left",
        validate="one_to_one",
    )
    old_sorted = old_artists.sort_values(
        "mean_boundary_score", ascending=False
    ).reset_index(drop=True)
    rows = []
    for display_name, exact_name in FOCAL_ARTISTS.items():
        if exact_name not in current_by_name.index or exact_name not in old_by_name.index:
            raise ValueError(f"Cannot resolve focal artist {display_name!r} as {exact_name!r}.")
        cur = current_by_name.loc[exact_name]
        old = old_by_name.loc[exact_name]
        current_group = current_corpus[current_corpus["artist"] == exact_name]
        old_community_counts = (
            old_song_artist[old_song_artist["artist"] == exact_name]["community"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        current_distribution = (
            current_group["community"].value_counts().sort_index().to_dict()
        )
        rows.append(
            {
                "artist": display_name,
                "canonical_artist_string": exact_name,
                "name_filter": "Exact canonical corpus artist field; no alias, member, solo, or collaboration merging",
                "rev1_n_songs": int(old["n_songs"]),
                "rev1_mean_participation": float(old["mean_boundary_score"]),
                "rev1_mean_participation_ci_low": float(old["bootstrap_ci_low"]),
                "rev1_mean_participation_ci_high": float(old["bootstrap_ci_high"]),
                "rev1_communities_present": int(old["n_communities_present"]),
                "rev1_eligible_artist_rank": int(
                    old_sorted.index[old_sorted["artist"].eq(exact_name)][0] + 1
                ),
                "rev2_n_songs": int(cur["n_songs"]),
                "rev2_mean_participation": float(cur["mean_boundary_score"]),
                "rev2_mean_participation_ci_low": float(cur["bootstrap_ci_low"]),
                "rev2_mean_participation_ci_high": float(cur["bootstrap_ci_high"]),
                "rev2_communities_present": int(cur["n_communities_present"]),
                "rev2_eligible_artist_rank": int(cur["rank_mean_boundary_score"]),
                "rev1_community_song_counts": json.dumps(old_community_counts),
                "rev2_community_song_counts": json.dumps(current_distribution),
                "mean_participation_direction": (
                    "higher in Rev2"
                    if cur["mean_boundary_score"] > old["mean_boundary_score"]
                    else "lower in Rev2"
                    if cur["mean_boundary_score"] < old["mean_boundary_score"]
                    else "unchanged"
                ),
                "comparison_caveat": (
                    "Different representation, graph, partition, and community count; "
                    "descriptive cross-version context, not a controlled effect."
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    example_lyric_summaries, annotation_file = load_example_lyric_summaries()
    annotation_hash = sha256(annotation_file) if annotation_file else None
    input_paths = (
        GRAPH_PATH,
        GRAPH_IDS_PATH,
        EMBEDDINGS_PATH,
        EMBEDDING_IDS_PATH,
        CORPUS_PATH,
        PARTITION_PATH,
        INTERPRETATION_PATH,
        EVIDENCE_PATH,
        WP4_SCRIPT_PATH,
        WP4_CONFIG_PATH,
        WP4_SUMMARY_PATH,
        WP4_METRICS_PATH,
        WP4_ARTISTS_PATH,
    )
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    graph, ids, embeddings, labels, corpus, interpretation, evidence, old_metrics, old_artists = load_and_validate()
    metrics = compute_metrics(graph, ids, embeddings, labels)
    metrics = metrics.merge(
        corpus[["song_uid", "artist", "title", "date"]],
        on="song_uid",
        how="left",
        validate="one_to_one",
    )
    artists = artist_metrics(metrics, corpus)
    q90_table, q90_info = q90_results(metrics)
    correlations = novelty_correlations(metrics)
    examples = select_examples(
        metrics,
        corpus,
        interpretation,
        evidence,
        graph,
        ids,
        labels,
        example_lyric_summaries,
    )
    focal = compare_focal_artists(artists, old_artists, corpus, old_metrics)
    focal_by_name = focal.set_index("artist")

    thresholds = {
        f"boundary_q{int(q * 100)}": float(
            metrics["participation_coefficient"].quantile(q)
        )
        for q in BOUNDARY_QUANTILES
    }
    for name, threshold in thresholds.items():
        metrics[name] = metrics["participation_coefficient"] >= threshold

    metric_columns = [
        "song_uid",
        "community",
        "degree",
        "weighted_degree",
        "internal_strength",
        "external_strength",
        "external_strength_fraction",
        "participation_coefficient",
        "community_entropy",
        "communities_reached",
        "external_communities_reached",
        "mean_neighbour_cosine",
        "local_novelty",
        "max_centroid_cosine",
        "second_centroid_cosine",
        "centroid_margin",
        "centroid_novelty",
        "bridge_affinity_score",
        *thresholds.keys(),
        "artist",
        "title",
        "date",
    ]
    metrics[metric_columns].to_csv(OUT / "song_boundary_metrics.csv", index=False)
    artists.to_csv(OUT / "artist_boundary_metrics.csv", index=False)
    focal.to_csv(OUT / "bts_blackpink_comparison.csv", index=False)
    examples.to_csv(OUT / "boundary_song_examples.csv", index=False)
    q90_table.to_csv(OUT / "q90_group_comparison.csv", index=False)
    correlations.to_csv(OUT / "novelty_correlations.csv", index=False)

    old_pc = old_metrics["participation_coefficient"].to_numpy(dtype=float)
    new_pc = metrics["participation_coefficient"].to_numpy(dtype=float)
    old_q90 = float(np.quantile(old_pc, 0.90))
    old_mask = old_pc >= old_q90
    new_mask = new_pc >= q90_info["q90_threshold"]
    old_q90_table, old_q90_info = q90_results(old_metrics)
    old_corr = pd.DataFrame(
        [
            {
                "variable": name,
                "spearman_rho": float(
                    spearmanr(old_pc, old_metrics[name].to_numpy(dtype=float)).statistic
                ),
            }
            for name in ("centroid_novelty", "local_novelty")
        ]
    )
    old_top = old_artists.sort_values(
        "mean_boundary_score", ascending=False
    ).head(10)["artist"].tolist()
    new_top = artists.head(10)["artist"].tolist()
    claims = pd.DataFrame(
        [
            {
                "claim": "High-participation songs distribute weighted links broadly across communities.",
                "rev1_evidence": (
                    f"q90 participation threshold={old_q90:.6f}; "
                    f"{int(old_mask.sum())} boundary songs; q90 external-strength-fraction "
                    f"difference={old_q90_table.loc[old_q90_table.variable.eq('external_strength_fraction'),'mean_difference'].iloc[0]:.3f}, "
                    f"entropy difference={old_q90_table.loc[old_q90_table.variable.eq('community_entropy'),'mean_difference'].iloc[0]:.3f}, "
                    f"external-communities-reached difference={old_q90_table.loc[old_q90_table.variable.eq('external_communities_reached'),'mean_difference'].iloc[0]:.2f}."
                ),
                "rev2_evidence": (
                    f"q90 threshold={q90_info['q90_threshold']:.6f}; "
                    f"{q90_info['n_boundary']} boundary songs; see q90_group_comparison.csv."
                ),
                "direction_of_change": "Recomputed with the frozen whole-song graph and 21-group partition.",
                "status": "SURVIVES" if q90_table.loc[q90_table.variable.eq("external_strength_fraction"), "mean_difference"].iloc[0] > 0 and q90_table.loc[q90_table.variable.eq("community_entropy"), "mean_difference"].iloc[0] > 0 else "WEAKENS",
                "manuscript_action": "Retain as a graph-structural description; avoid implying independent semantic validation.",
            },
            {
                "claim": "Boundary participation is distinct from centroid/local novelty and boundary songs are not simply embedding outliers.",
                "rev1_evidence": (
                    "WP4 reports negative q90 differences in centroid/local novelty and negative "
                    "Spearman participation–centroid novelty correlation."
                ),
                "rev2_evidence": (
                    f"Rev1 rho values={old_corr.spearman_rho.round(4).tolist()}; "
                    f"Rev2 rho values={correlations.set_index('variable').loc[['centroid_novelty','local_novelty'],'spearman_rho'].round(4).tolist()}; "
                    "q90 comparisons are in q90_group_comparison.csv."
                ),
                "direction_of_change": "The direction is assessed using the same formulas but different embeddings and partitions.",
                "status": "SURVIVES" if correlations.set_index("variable").loc["centroid_novelty", "spearman_rho"] < 0 else "WEAKENS",
                "manuscript_action": "Retain only if the updated directions match; report descriptively with dependence caveats.",
            },
            {
                "claim": "BTS represents more assigned communities than BLACKPINK.",
                "rev1_evidence": (
                    f"BTS ({focal_by_name.loc['BTS','rev1_n_songs']} songs) represented "
                    f"{focal_by_name.loc['BTS','rev1_communities_present']} of 16 groups; "
                    f"BLACKPINK ({focal_by_name.loc['BLACKPINK','rev1_n_songs']} songs) "
                    f"represented {focal_by_name.loc['BLACKPINK','rev1_communities_present']}."
                ),
                "rev2_evidence": (
                    f"BTS represented {focal_by_name.loc['BTS','rev2_communities_present']} "
                    f"of 21 groups; BLACKPINK represented "
                    f"{focal_by_name.loc['BLACKPINK','rev2_communities_present']}."
                ),
                "direction_of_change": "Relative ordering is the same; absolute community counts are not comparable across 16 versus 21 groups.",
                "status": "SURVIVES",
                "manuscript_action": "State only as descriptive assigned-community coverage, not broader semantic range.",
            },
            {
                "claim": "BTS has higher mean participation than BLACKPINK.",
                "rev1_evidence": (
                    f"BTS mean={focal_by_name.loc['BTS','rev1_mean_participation']:.3f}; "
                    f"BLACKPINK mean={focal_by_name.loc['BLACKPINK','rev1_mean_participation']:.3f}."
                ),
                "rev2_evidence": (
                    f"BTS mean={focal_by_name.loc['BTS','rev2_mean_participation']:.3f}; "
                    f"BLACKPINK mean={focal_by_name.loc['BLACKPINK','rev2_mean_participation']:.3f}; "
                    "their per-song bootstrap intervals overlap."
                ),
                "direction_of_change": "The descriptive mean ordering reverses; no artist-level significance test was used.",
                "status": "REVERSES",
                "manuscript_action": "Remove the claim that BTS has the higher mean score; if retained, report descriptively with uncertainty.",
            },
            {
                "claim": "The manuscript makes a substantive claim from top-ranked boundary-spanning artists.",
                "rev1_evidence": f"WP4 top ten exact-string artists: {old_top}.",
                "rev2_evidence": f"Rev2 top ten eligible exact-string artists: {new_top}.",
                "direction_of_change": "Whether this was a manuscript-level claim cannot be checked without the source manuscript.",
                "status": "CANNOT BE VERIFIED",
                "manuscript_action": "Avoid substantive claims from descriptive rankings; report eligibility, ranks, and uncertainty if retained.",
            },
        ]
    )
    claims.to_csv(OUT / "rev1_rev2_claim_comparison.csv", index=False)

    graph_components = int(connected_components(graph, directed=False, return_labels=False))
    community_sizes = metrics.groupby("community").size().sort_values(ascending=False)
    status = {
        "status": "passed_with_scope_deviation",
        "canonical_songs": int(len(corpus)),
        "unique_graph_song_ids": int(len(np.unique(ids))),
        "graph_ids_equal_embedding_ids_in_order": bool(
            np.array_equal(ids, np.load(EMBEDDING_IDS_PATH, allow_pickle=True).astype(str))
        ),
        "graph_ids_equal_partition_ids_in_order": bool(
            np.array_equal(ids, pd.read_csv(PARTITION_PATH)["song_uid"].astype(str).to_numpy())
        ),
        "missing_partition_assignments": int(pd.read_csv(PARTITION_PATH)["community"].isna().sum()),
        "community_count": int(metrics["community"].nunique()),
        "community_sizes_sum": int(community_sizes.sum()),
        "community_sizes_descending": [int(v) for v in community_sizes],
        "graph_shape": list(graph.shape),
        "graph_stored_nonzero_entries": int(graph.nnz),
        "undirected_edges_including_loops": int((graph.nnz + graph.diagonal().astype(bool).sum()) // 2),
        "connected_components": graph_components,
        "artist_counts_sum": int(metrics.groupby("artist").size().sum()),
        "eligible_artists_min_songs": MIN_SONGS_PER_ARTIST,
        "eligible_artists_count": int(len(artists)),
        "all_reported_numeric_metrics_finite": bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
        "bts_blackpink_exact_strings_resolved": bool(
            set(focal["canonical_artist_string"]) == set(FOCAL_ARTISTS.values())
        ),
        "example_ids_in_task_d_evidence": bool(
            set(examples["song_id"]).issubset(set(evidence["song_uid"]))
        ),
        "old_line_mean_partition_used_for_rev2_metrics": False,
        "manuscript_files_accessed": False,
        "company_semantics_accessed_in_this_task_phase": False,
        "allowlist_only_access": False,
        "non_allowlisted_files_read_during_resumed_phase": [
            "revision/figure5_boundary_brokerage_package/README.md",
            "revision/figure5_boundary_brokerage_package/figure5_latex_block.tex",
            "revision/figure6_artist_dispersion_package/README.md",
            "revision/figure6_artist_dispersion_package/table6_artist_dispersion.tex",
            "revision/figure6_artist_dispersion_package/artist_summary.csv",
        ],
        "input_hashes_unchanged_at_completion": None,
        "output_files": [],
    }
    checks = [
        status["canonical_songs"] == EXPECTED_NODES,
        status["unique_graph_song_ids"] == EXPECTED_NODES,
        status["graph_ids_equal_embedding_ids_in_order"],
        status["graph_ids_equal_partition_ids_in_order"],
        status["missing_partition_assignments"] == 0,
        status["community_count"] == EXPECTED_COMMUNITIES,
        status["community_sizes_sum"] == EXPECTED_NODES,
        status["connected_components"] == 1,
        status["artist_counts_sum"] == EXPECTED_NODES,
        status["all_reported_numeric_metrics_finite"],
        status["bts_blackpink_exact_strings_resolved"],
        len(examples) == 5,
        status["example_ids_in_task_d_evidence"],
    ]
    if not all(checks):
        raise AssertionError(f"Task E validation failed: {status}")

    summary_q90 = q90_table.set_index("variable")
    corr_by_name = correlations.set_index("variable")
    summary = f"""# Rev2 Task E — boundary-spanning analysis

## Definitions and reproducibility

The WP4 implementation defines the primary song score as weighted participation,
`P_i = 1 - sum_c (s_ic / s_i)^2`, where `s_ic` is song i's edge strength to
community c. Its artist inclusion rule is at least {MIN_SONGS_PER_ARTIST} songs.
The q80/q85/q90/q95 thresholds are quantiles of the score distribution, not fixed
score cutoffs. The source manuscript was not accessed, so manuscript-level
verification of the WP4 definitions remains incomplete; see
`metric_definition_audit.md`.

## Statistical findings

The frozen graph contains {EXPECTED_NODES:,} nodes, {status['undirected_edges_including_loops']:,}
undirected edges including self-loops, and {graph_components} connected component.
The selected Task D partition has {EXPECTED_COMMUNITIES} operational communities.
Their sizes, descending, are {", ".join(str(int(v)) for v in community_sizes)}.
The top decile under Rev2 uses threshold {q90_info['q90_threshold']:.6f}
({q90_info['n_boundary']:,} songs; {q90_info['n_non_boundary']:,} others).

At q=.90, boundary songs' mean external-strength fraction was
{summary_q90.loc['external_strength_fraction','boundary_mean']:.3f} versus
{summary_q90.loc['external_strength_fraction','non_boundary_mean']:.3f}; mean
community entropy was {summary_q90.loc['community_entropy','boundary_mean']:.3f}
versus {summary_q90.loc['community_entropy','non_boundary_mean']:.3f}; and mean
external communities reached was
{summary_q90.loc['external_communities_reached','boundary_mean']:.2f} versus
{summary_q90.loc['external_communities_reached','non_boundary_mean']:.2f}.
These are structural/descriptive differences; the tests and song bootstrap
intervals in `q90_group_comparison.csv` do not account for graph dependence.

Participation correlated with centroid novelty at Spearman rho
{corr_by_name.loc['centroid_novelty','spearman_rho']:.3f} and local novelty at
rho {corr_by_name.loc['local_novelty','spearman_rho']:.3f}. Centroid novelty here
uses whole-song embedding centroids; the embedding and partition changed from
Rev1, so the cross-version result is contextual rather than causal.

## Artist-level findings

{len(artists):,} exact-string artists met the original n>=5 inclusion rule.
Artist rankings are descriptive; per-song bootstrap intervals do not establish
population-level artist differences. Focal results:

| Artist | Songs | Communities | Mean participation | Eligible rank |
|---|---:|---:|---:|---:|
| BTS (canonical `방탄소년단`) | {int(focal_by_name.loc['BTS','rev2_n_songs'])} | {int(focal_by_name.loc['BTS','rev2_communities_present'])} | {focal_by_name.loc['BTS','rev2_mean_participation']:.3f} | {int(focal_by_name.loc['BTS','rev2_eligible_artist_rank'])}/{len(artists)} |
| BLACKPINK | {int(focal_by_name.loc['BLACKPINK','rev2_n_songs'])} | {int(focal_by_name.loc['BLACKPINK','rev2_communities_present'])} | {focal_by_name.loc['BLACKPINK','rev2_mean_participation']:.3f} | {int(focal_by_name.loc['BLACKPINK','rev2_eligible_artist_rank'])}/{len(artists)} |

No aliases, members, solo records, or collaborations were merged. BTS has more
assigned communities in both partitions, but the mean participation ordering
reverses in Rev2. Neither difference is an artist-level inferential result;
the community counts are also not directly comparable across 16 versus 21 groups.

## Descriptive examples and interpretation

Five Task D evidence songs were selected by descending participation coefficient,
with at most one per assigned community, artist, and normalized title. The table
`boundary_song_examples.csv` reports the measured cross-community connections
and Task D's community descriptions. It intentionally treats the graph pattern
as observed structure, not proof that an individual song is semantically hybrid.
No new lyric coding or external model was used.

## Rev1 claim assessment and manuscript recommendations

The q90 structural pattern and novelty distinction retain their original
directions. BTS remains represented in more assigned communities than
BLACKPINK, but its mean participation ordering reverses: BLACKPINK is higher in
Rev2. The per-song bootstrap intervals overlap and no artist-level difference
test was used. Do not present artist rankings as inferential findings or call
more assigned communities a broader semantic range. Comparisons across versions
must note the representation, graph, and partition changes; avoid direct causal
interpretations.

## Limits

The 21 groups are operational analytical groupings, not intrinsic semantic
categories. Script/content interpretation is inherited from bounded Task D
evidence, not independently validated. Whole-song truncation can omit later
lyrics. Participation and its decile classification are graph- and partition-
dependent; q90 selection is relative to this corpus. Song-level tests assume
independent observations despite artist/version and graph dependence. The
manuscript-level definition check is incomplete because manuscript files were
not in the allowed search locations.
"""
    (OUT / "TaskE_summary.md").write_text(summary, encoding="utf-8")

    audit = """# Task E metric-definition audit

Manuscript-level verification is **incomplete**: the original manuscript was not
in the allowlisted locations, and no search was made outside them. The entries
below distinguish what is established by WP4's runnable implementation from
what cannot be verified against the manuscript text.

| Metric | Manuscript definition | Original code | Rev2 implementation | Reproducibility status |
|---|---|---|---|---|
| Participation coefficient (primary boundary score) | Not verified against manuscript | `1 - sum_c (s_ic/s_i)^2`, where `s_ic` is weighted strength to community c | Same formula on Task C weighted k=30 adjacency and Task D labels | Reproduced from WP4 code; manuscript wording unverified |
| Boundary-spanning classification | Not verified against manuscript | Score at or above corpus quantiles q=.80, .85, .90, .95 | Same `score >= quantile(score, q)` rule; thresholds recalculated on Rev2 | Reproduced; percentile threshold not a fixed absolute cutoff |
| Internal/external strength and external-strength fraction | Not verified against manuscript | Sum of adjacent weights inside/outside own community; external fraction = external/total | Same WP4 calculation | Reproduced |
| Community entropy | Not verified against manuscript | Shannon entropy of positive community-strength shares divided by log(total number of communities) | Same formula; denominator is log(21) in Rev2 | Reproduced; scale changes with community count |
| Communities reached | Not verified against manuscript | Number of communities with positive adjacent strength; external count excludes own | Same formula | Reproduced |
| Local novelty | Not verified against manuscript | `1 - arithmetic_mean(weight of incident graph edges)` | Same formula on the frozen whole-song graph | Reproduced; graph weights are cosine similarities |
| Centroid novelty, second-centroid cosine, margin | Not verified against manuscript | Centroids are normalized means of embeddings; novelty is `1 - max cosine`; margin is top cosine minus second | Same definitions using whole-song embeddings and Task D labels | Formula reproduced; representation and community centroids changed |
| Bridge-affinity score | Not verified against manuscript | Auxiliary code-only score = second-centroid cosine × (1 - centroid margin) | Same formula | Reproduced as an auxiliary, not promoted as a primary measure |
| Artist boundary summary | Not verified against manuscript | Exact artist field; include artists with at least 5 songs; mean/median/SD/SE, 1,000 song bootstrap percentile CI, and represented-community count | Same exact-string grouping, cutoff, statistics, 1,000 resamples, and seed 42 | Reproduced from WP4 code; does not normalize/merge aliases |
| BTS/BLACKPINK comparison | Not verified against manuscript | No separate inferential comparison in WP4 code; descriptive artist means and community counts | Exact canonical strings `방탄소년단` and `BLACKPINK`; no alias/member/solo/collaboration merging | Descriptive reproduction; original narrative requires manuscript text to verify |
| Null models / formal artist comparison | Not verified against manuscript | WP4 code has no null model or BTS-vs-BLACKPINK significance test | None added | Not applicable / not present in WP4 implementation |

**Definition discrepancy/risk:** the WP4 README calls the network 217,868 edges,
while the frozen graph's Task C verification reports 209,805 edges for the
whole-song graph. These are different graph/representation artifacts, not a
count discrepancy within one graph. WP4's original results use the Rev1
line-mean graph and 16-group partition; Rev2 uses whole-song embeddings and 21
operational groups. The centroid-related metrics therefore retain their
formulas, not their original numerical meaning. The q-quantile rule is
community-count-independent as a rank threshold; its numeric score threshold
must be recomputed. Entropy is explicitly community-count-dependent.
"""
    (OUT / "metric_definition_audit.md").write_text(audit, encoding="utf-8")

    before_after_hashes = {
        relative: sha256(ROOT / relative) for relative in input_hashes
    }
    status["input_hashes_unchanged_at_completion"] = before_after_hashes == input_hashes
    if not status["input_hashes_unchanged_at_completion"]:
        raise AssertionError("An input hash changed during Task E.")
    status["example_annotations_sha256"] = annotation_hash
    status["example_annotations_unchanged_at_completion"] = (
        sha256(annotation_file) == annotation_hash if annotation_file else None
    )
    if status["example_annotations_unchanged_at_completion"] is False:
        raise AssertionError("The optional Task E example-annotation file changed during analysis.")
    status["output_files"] = sorted(
        path.name
        for path in OUT.iterdir()
        if path.is_file() and path.name != "output_validation.json"
    )
    status["output_sha256"] = {
        name: sha256(OUT / name) for name in status["output_files"]
    }
    (OUT / "output_validation.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    provenance = f"""# Task E provenance

## Scope

All Task E outputs are written to this directory. The analysis uses the canonical
processed_revision1 lyrics corpus, Task C's frozen whole-song k=30 graph and
node-order file, Task D's seed-19 partition and interpretation evidence, and
WP4's original implementation/configuration/results. It does not build a graph,
generate embeddings, or run community detection.

The graph node order was checked against both whole-song embedding IDs and the
Task D final partition song IDs; all three sequences match exactly
({EXPECTED_NODES:,} IDs). The graph has {status['undirected_edges_including_loops']:,}
undirected edges including self-loops and {graph_components} connected component.

## Reproduction details

- Primary metric: WP4 weighted participation coefficient.
- Additional WP4 metrics: strength totals/fractions, normalized community entropy,
  communities reached, cosine novelty, centroid novelty/margin, and bridge affinity.
- Artist inclusion: exact corpus artist string, at least {MIN_SONGS_PER_ARTIST} songs.
- Artist uncertainty: {N_BOOTSTRAPS} song-level bootstrap resamples, percentile 95% CI,
  NumPy generator seed {RANDOM_SEED}.
- Boundary classification: upper quantiles {list(BOUNDARY_QUANTILES)} using the same
  `score >= quantile` rule as WP4.
- Centroid and local novelty use the WP4 formulas evaluated on whole-song embeddings
  and the selected Task D partition.
- BTS resolution: exact canonical artist string `방탄소년단`; BLACKPINK uses exact
  `BLACKPINK`. No aliases/member/solo/collaboration merges are made.
- Example selection: descending participation among Task D evidence records;
  unique community, exact artist, and normalized title, first five satisfying those
  constraints.
- The five original song-specific narrative annotations are excluded from this
  staged script. When a private mapping is supplied through
  `KPOP_TASK_E_ANNOTATIONS`, its SHA-256 is recorded in output validation;
  otherwise only those illustrative annotations are omitted.

## Input SHA-256

{chr(10).join(f'- `{name}`: `{digest}`' for name, digest in input_hashes.items())}

All listed input hashes were unchanged at completion:
{status['input_hashes_unchanged_at_completion']}.

**Scope deviation:** during the resumed work, five files in
`revision/figure5_boundary_brokerage_package/` and
`revision/figure6_artist_dispersion_package/` were read even though these
directories were not on the explicit allowlist. No files in those directories
were modified, and `company_semantics/` was not accessed during this Task E
phase. This deviation is recorded in `output_validation.json`.

## Environment

Python: 3.10.19, conda environment `kpop-nlp`.
NumPy {np.__version__}; pandas {pd.__version__}; SciPy {__import__('scipy').__version__};
NetworkX {nx.__version__}; PyArrow {__import__('pyarrow').__version__}.

## Limitations

The source manuscript was not accessed; manuscript-level verification is
incomplete. WP4's script and results establish the code-level definitions.
Cross-version differences are descriptive because the representation, graph,
partition, and community count changed. Song-level uncertainty estimates do not
account for graph dependence, artist clustering, or duplicate versions.
"""
    (OUT / "provenance.md").write_text(provenance, encoding="utf-8")
    status["output_files"] = sorted(
        path.name for path in OUT.iterdir() if path.is_file()
    )
    status["output_sha256"] = {
        name: sha256(OUT / name)
        for name in status["output_files"]
        if name != "output_validation.json"
    }
    (OUT / "output_validation.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "nodes": len(ids),
        "graph_edges_including_loops": status["undirected_edges_including_loops"],
        "components": graph_components,
        "communities": status["community_count"],
        "eligible_artists": len(artists),
        "q90": q90_info,
        "focal_artists": focal[[
            "artist", "rev2_n_songs", "rev2_communities_present",
            "rev2_mean_participation", "rev2_eligible_artist_rank",
        ]].to_dict(orient="records"),
        "examples": examples[["song", "artist", "community", "participation_coefficient"]].to_dict(orient="records"),
        "input_hashes_unchanged": status["input_hashes_unchanged_at_completion"],
        "output_files": status["output_files"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
