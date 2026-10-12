"""Regenerate Rev2 community characterization on the selected whole-song partition."""

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
from scipy import sparse
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(os.environ.get("KPOP_PROJECT_ROOT", Path(__file__).resolve().parents[4])).expanduser().resolve()
OUT = Path(os.environ.get("KPOP_TASK_D_OUTPUT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
DATA = ROOT / "data/processed_revision1"
TASK_B = Path(os.environ.get("KPOP_TASK_B_OUTPUT_DIR", ROOT / "revision/Rev2_TaskB_representation_comparison")).expanduser().resolve()
TASK_C = Path(os.environ.get("KPOP_TASK_C_OUTPUT_DIR", ROOT / "revision/Rev2_TaskC_alternative_communities")).expanduser().resolve()
PERMUTATIONS = 999
SEED = 42
STRATA = [
    "hangul_dominant",
    "mixed_hangul_latin",
    "latin_dominant",
    "other_script",
    "no_letters",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    with (OUT / name).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def write_json(value: object, name: str) -> None:
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]", " ", str(value))).strip()


def script_counts(text: str) -> tuple[int, int, int, int, float, float, float, str]:
    hangul = latin = other = 0
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        if "HANGUL" in name:
            hangul += 1
        elif "LATIN" in name:
            latin += 1
        else:
            other += 1
    total = hangul + latin + other
    if total == 0:
        return hangul, latin, other, total, np.nan, np.nan, np.nan, "no_letters"
    hangul_share, latin_share, other_share = (
        hangul / total,
        latin / total,
        other / total,
    )
    stratum = (
        "hangul_dominant"
        if hangul_share >= 0.8
        else "latin_dominant"
        if latin_share >= 0.8
        else "mixed_hangul_latin"
        if hangul_share + latin_share >= 0.8
        else "other_script"
    )
    return (
        hangul,
        latin,
        other,
        total,
        hangul_share,
        latin_share,
        other_share,
        stratum,
    )


def json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def main() -> None:
    if any((OUT / name).exists() for name in (
        "final_partition.csv",
        "partition_selection.csv",
        "community_profiles.csv",
        "community_representative_songs.csv",
        "community_terms.csv",
        "community_script_summary.csv",
        "script_association.csv",
        "selected_interpretation_evidence.csv",
        "analysis_metadata.json",
    )):
        raise FileExistsError("Task D numeric outputs already exist; refusing to overwrite.")

    embedding_path = DATA / "lyrics_embeddings_whole_song_truncated.npy"
    embedding_ids_path = DATA / "lyrics_embedding_ids_whole_song_truncated.npy"
    corpus_path = DATA / "lyrics.parquet"
    ids = np.load(TASK_C / "song_ids.npy", allow_pickle=False).astype(str)
    labels = np.load(TASK_C / "baseline_labels.npy", allow_pickle=False)
    embeddings = np.load(embedding_path, allow_pickle=False)
    embedding_ids = np.load(embedding_ids_path, allow_pickle=True).astype(str)
    runs = pd.read_csv(TASK_B / "louvain_repeated_runs.csv")
    whole_runs = runs.loc[runs["representation"] == "whole_song_truncated"].copy()
    whole_runs = whole_runs.sort_values("seed").reset_index(drop=True)

    if labels.shape != (20, 7891) or len(ids) != 7891 or len(set(ids)) != 7891:
        raise ValueError("Expected 20 saved partitions and 7,891 unique ordered song IDs.")
    if embeddings.shape != (7891, 384) or len(embedding_ids) != 7891:
        raise ValueError("Unexpected whole-song embedding dimensions or ID count.")
    if not np.array_equal(ids, embedding_ids):
        raise ValueError("Task C graph IDs do not exactly match embedding ID order.")
    if len(whole_runs) != 20 or whole_runs["seed"].tolist() != list(range(20)):
        raise ValueError("Task B does not contain the expected whole-song seeds 0–19.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Whole-song embeddings contain non-finite values.")
    if any(
        not np.array_equal(np.unique(partition), np.arange(np.unique(partition).size))
        for partition in labels
    ):
        raise ValueError("Saved partition community labels are not contiguous within seed.")

    corpus = pd.read_parquet(corpus_path)
    if (
        len(corpus) != 7891
        or corpus["song_uid"].isna().any()
        or corpus["song_uid"].duplicated().any()
        or set(corpus["song_uid"].astype(str)) != set(ids)
    ):
        raise ValueError("Canonical corpus IDs do not match the saved partition IDs.")
    if corpus["lyrics"].isna().any() or not corpus["lyrics"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Canonical lyric text is missing or non-string.")

    adjacency = sparse.load_npz(TASK_C / "graph.npz")
    if adjacency.shape != (7891, 7891):
        raise ValueError("Task C graph has unexpected dimensions.")
    graph = nx.from_scipy_sparse_array(adjacency, edge_attribute="weight")
    graph = nx.relabel_nodes(graph, dict(enumerate(ids)), copy=False)
    if graph.number_of_nodes() != 7891 or graph.number_of_edges() != 209805:
        raise ValueError("Task C graph does not match the expected canonical k=30 graph.")
    if nx.number_connected_components(graph) != 1:
        raise ValueError("Expected the whole-song graph to have one component.")

    pairwise_nmi = np.eye(20, dtype=float)
    pairwise_ari = np.eye(20, dtype=float)
    for first in range(20):
        for second in range(first + 1, 20):
            pairwise_nmi[first, second] = pairwise_nmi[second, first] = (
                normalized_mutual_info_score(labels[first], labels[second])
            )
            pairwise_ari[first, second] = pairwise_ari[second, first] = (
                adjusted_rand_score(labels[first], labels[second])
            )
    mean_nmi = (pairwise_nmi.sum(axis=1) - 1) / 19
    mean_ari = (pairwise_ari.sum(axis=1) - 1) / 19

    run_modularity = whole_runs.set_index("seed")["modularity"].astype(float)
    run_counts = whole_runs.set_index("seed")["n_communities"].astype(int)
    partition_rows = []
    for seed, partition in enumerate(labels):
        groups, counts = np.unique(partition, return_counts=True)
        if len(groups) != run_counts.loc[seed]:
            raise ValueError(f"Saved partition count differs from Task B at seed {seed}.")
        sets = [set(ids[partition == group]) for group in groups]
        modularity = nx.community.modularity(graph, sets, weight="weight", resolution=1)
        if abs(modularity - run_modularity.loc[seed]) > 1e-10:
            raise ValueError(f"Saved partition modularity differs from Task B at seed {seed}.")
        partition_rows.append(
            {
                "seed": seed,
                "community_count": len(groups),
                "task_b_reported_modularity": float(run_modularity.loc[seed]),
                "standard_modularity_evaluation_gamma_1": float(modularity),
                "mean_nmi_to_other_seeds": float(mean_nmi[seed]),
                "mean_ari_to_other_seeds": float(mean_ari[seed]),
                "community_sizes_descending": json_compact(
                    sorted(counts.astype(int).tolist(), reverse=True)
                ),
                "selected": False,
            }
        )
    selected_seed = min(
        range(20),
        key=lambda seed: (-mean_nmi[seed], -run_modularity.loc[seed], seed),
    )
    partition_rows[selected_seed]["selected"] = True
    selected_labels = labels[selected_seed].astype(int)
    write_csv(pd.DataFrame(partition_rows), "partition_selection.csv")
    write_csv(
        pd.DataFrame(
            {"song_uid": ids, "community": selected_labels, "selected_seed": selected_seed}
        ),
        "final_partition.csv",
    )

    norms = np.linalg.norm(embeddings.astype(float), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Embeddings contain zero or invalid vectors.")
    normalized = embeddings.astype(float) / norms[:, None]
    community_codes = selected_labels
    communities = np.unique(community_codes)
    ordered = (
        corpus.assign(song_uid=corpus["song_uid"].astype(str))
        .set_index("song_uid")
        .loc[ids]
        .reset_index()
    )
    ordered["community"] = community_codes
    texts = ordered["lyrics"].map(clean_text)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        min_df=3,
        max_df=0.90,
        max_features=30000,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w\w+\b",
    )
    tfidf = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    profiles: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    for community in communities:
        indices = np.flatnonzero(community_codes == community)
        inside = np.asarray(tfidf[indices].mean(axis=0)).ravel()
        outside = np.asarray(tfidf[community_codes != community].mean(axis=0)).ravel()
        score = inside - outside
        top_terms = np.lexsort((terms, -score))[:20]
        for rank, term_index in enumerate(top_terms, start=1):
            term_rows.append(
                {
                    "community": int(community),
                    "rank": rank,
                    "term": terms[term_index],
                    "distinctive_tfidf": float(score[term_index]),
                    "inside_mean_tfidf": float(inside[term_index]),
                    "outside_mean_tfidf": float(outside[term_index]),
                    "inside_document_count": int((tfidf[indices, term_index] > 0).sum()),
                }
            )

        center = normalized[indices].mean(axis=0)
        center /= np.linalg.norm(center)
        cosine = normalized[indices] @ center
        subgraph = graph.subgraph(ids[indices])
        pagerank_map = nx.pagerank(
            subgraph, alpha=0.85, weight="weight", tol=1e-6, max_iter=100
        )
        pagerank = np.asarray([pagerank_map[song_id] for song_id in ids[indices]])
        centroid_order = np.lexsort((ids[indices], -cosine))
        pagerank_order = np.lexsort((ids[indices], -pagerank))
        centroid_rank = np.empty(len(indices), dtype=int)
        pagerank_rank = np.empty(len(indices), dtype=int)
        centroid_rank[centroid_order] = np.arange(1, len(indices) + 1)
        pagerank_rank[pagerank_order] = np.arange(1, len(indices) + 1)

        for local_index in np.flatnonzero((centroid_rank <= 10) | (pagerank_rank <= 10)):
            row = ordered.iloc[indices[local_index]]
            representative_rows.append(
                {
                    "community": int(community),
                    "song_uid": row["song_uid"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "centroid_rank": int(centroid_rank[local_index]),
                    "cosine_to_centroid": float(cosine[local_index]),
                    "pagerank_rank": int(pagerank_rank[local_index]),
                    "within_community_pagerank": float(pagerank[local_index]),
                    "within_community_weighted_degree": float(
                        subgraph.degree(row["song_uid"], weight="weight")
                    ),
                    "selection": (
                        "both"
                        if centroid_rank[local_index] <= 10 and pagerank_rank[local_index] <= 10
                        else "centroid"
                        if centroid_rank[local_index] <= 10
                        else "pagerank"
                    ),
                    "lyric_snippet": texts.iloc[indices[local_index]][:180]
                    + ("..." if len(texts.iloc[indices[local_index]]) > 180 else ""),
                }
            )

        centroid_rows = [
            {
                "song_uid": ordered.iloc[indices[local_index]]["song_uid"],
                "title": ordered.iloc[indices[local_index]]["title"],
                "artist": ordered.iloc[indices[local_index]]["artist"],
            }
            for local_index in centroid_order[:5]
        ]
        pagerank_rows = [
            {
                "song_uid": ordered.iloc[indices[local_index]]["song_uid"],
                "title": ordered.iloc[indices[local_index]]["title"],
                "artist": ordered.iloc[indices[local_index]]["artist"],
            }
            for local_index in pagerank_order[:5]
        ]
        artists = ordered.iloc[indices]["artist"].fillna("Unknown").value_counts()
        profiles.append(
            {
                "community": int(community),
                "n_songs": len(indices),
                "corpus_share": len(indices) / len(ordered),
                "n_artists": int(artists.size),
                "top_artist_share": float(artists.iloc[0] / len(indices)),
                "artist_hhi": float(((artists / len(indices)) ** 2).sum()),
                "top_artists": json_compact(
                    [
                        {"artist": artist, "n_songs": int(count), "share": count / len(indices)}
                        for artist, count in artists.head(10).items()
                    ]
                ),
                "mean_cosine_to_centroid": float(cosine.mean()),
                "median_cosine_to_centroid": float(np.median(cosine)),
                "q25_cosine_to_centroid": float(np.quantile(cosine, 0.25)),
                "q75_cosine_to_centroid": float(np.quantile(cosine, 0.75)),
                "top_terms": json_compact(terms[top_terms].tolist()),
                "top_centroid_songs": json_compact(centroid_rows),
                "top_pagerank_songs": json_compact(pagerank_rows),
                "representative_snippets": json_compact(
                    [
                        {
                            **centroid_rows[position],
                            "snippet": texts.iloc[indices[local_index]][:180],
                        }
                        for position, local_index in enumerate(centroid_order[:5])
                    ]
                ),
            }
        )

        centroid_list = list(centroid_order)
        pagerank_list = list(pagerank_order)
        selected_evidence: list[int] = []
        seen_ids: set[str] = set()
        for position in range(max(len(centroid_list), len(pagerank_list))):
            for ranking in (centroid_list, pagerank_list):
                if position >= len(ranking):
                    continue
                local_index = ranking[position]
                song_id = ids[indices[local_index]]
                if song_id not in seen_ids:
                    selected_evidence.append(local_index)
                    seen_ids.add(song_id)
                if len(selected_evidence) == 12:
                    break
            if len(selected_evidence) == 12:
                break
        for evidence_rank, local_index in enumerate(selected_evidence, start=1):
            ordered_row = ordered.iloc[indices[local_index]]
            evidence_rows.append(
                {
                    "community": int(community),
                    "evidence_rank": evidence_rank,
                    "song_uid": ordered_row["song_uid"],
                    "title": ordered_row["title"],
                    "artist": ordered_row["artist"],
                    "centroid_rank": int(centroid_rank[local_index]),
                    "pagerank_rank": int(pagerank_rank[local_index]),
                    "selection": (
                        "both"
                        if centroid_rank[local_index] <= 10 and pagerank_rank[local_index] <= 10
                        else "centroid"
                        if centroid_rank[local_index] <= 10
                        else "pagerank"
                    ),
                    "lyric_snippet": texts.iloc[indices[local_index]][:180]
                    + ("..." if len(texts.iloc[indices[local_index]]) > 180 else ""),
                }
            )

    write_csv(pd.DataFrame(profiles), "community_profiles.csv")
    write_csv(pd.DataFrame(representative_rows), "community_representative_songs.csv")
    write_csv(pd.DataFrame(term_rows), "community_terms.csv")
    write_csv(pd.DataFrame(evidence_rows), "selected_interpretation_evidence.csv")

    script_columns = [
        "hangul_characters",
        "latin_characters",
        "other_letter_characters",
        "letter_characters",
        "hangul_character_share",
        "latin_character_share",
        "other_letter_share",
        "script_stratum",
    ]
    script_data = pd.DataFrame(
        [script_counts(text) for text in ordered["lyrics"]], columns=script_columns
    )
    scripts = pd.concat(
        [ordered[["song_uid", "community"]].reset_index(drop=True), script_data], axis=1
    )
    table = pd.crosstab(scripts["community"], scripts["script_stratum"]).reindex(
        index=communities, columns=STRATA, fill_value=0
    )
    script_summary_rows = []
    for community, group in scripts.groupby("community", sort=True):
        row: dict[str, object] = {"community": int(community), "n_songs": len(group)}
        for column in (
            "hangul_character_share",
            "latin_character_share",
            "other_letter_share",
        ):
            values = group[column]
            for statistic, value in (
                ("mean", values.mean()),
                ("std", values.std()),
                ("q25", values.quantile(0.25)),
                ("median", values.median()),
                ("q75", values.quantile(0.75)),
                ("min", values.min()),
                ("max", values.max()),
            ):
                row[f"{column}_{statistic}"] = (
                    float(value) if pd.notna(value) else None
                )
        for stratum in STRATA:
            count = int(table.loc[community, stratum])
            row[f"{stratum}_n"] = count
            row[f"{stratum}_proportion"] = count / len(group)
        script_summary_rows.append(row)
    write_csv(pd.DataFrame(script_summary_rows), "community_script_summary.csv")

    test_table = table.loc[:, table.sum(axis=0) > 0]
    chi_square, asymptotic_p, degrees_freedom, expected = chi2_contingency(
        test_table.to_numpy(), correction=False
    )
    n_songs = len(scripts)
    rows, columns = test_table.shape
    phi = chi_square / n_songs
    corrected_phi = max(
        0.0, phi - (columns - 1) * (rows - 1) / (n_songs - 1)
    )
    cramers_v = float(np.sqrt(phi / min(rows - 1, columns - 1)))
    corrected_v = float(
        np.sqrt(
            corrected_phi
            / min(
                rows - 1 - (rows - 1) ** 2 / (n_songs - 1),
                columns - 1 - (columns - 1) ** 2 / (n_songs - 1),
            )
        )
    )
    _, community_codes = np.unique(scripts["community"], return_inverse=True)
    tested_strata = test_table.columns.tolist()
    stratum_codes = scripts["script_stratum"].map(
        {stratum: index for index, stratum in enumerate(tested_strata)}
    ).to_numpy()
    rng = np.random.default_rng(SEED)
    extreme = 0
    for _ in range(PERMUTATIONS):
        permuted = np.bincount(
            community_codes * columns + rng.permutation(stratum_codes),
            minlength=rows * columns,
        ).reshape(rows, columns)
        statistic = ((permuted - expected) ** 2 / expected).sum()
        extreme += int(statistic >= chi_square - 1e-12)

    continuous_effects: dict[str, dict[str, float | int]] = {}
    for column in ("hangul_character_share", "latin_character_share"):
        valid = scripts[column].notna().to_numpy()
        values = scripts.loc[valid, column].to_numpy()
        codes = community_codes[valid]
        sizes = np.bincount(codes, minlength=rows)
        mean = values.mean()
        total_ss = ((values - mean) ** 2).sum()

        def eta_squared(candidate: np.ndarray) -> float:
            sums = np.bincount(codes, weights=candidate, minlength=rows)
            between_ss = np.sum((sums - sizes * mean) ** 2 / np.maximum(sizes, 1))
            return float(between_ss / total_ss)

        observed = eta_squared(values)
        null = np.asarray(
            [eta_squared(rng.permutation(values)) for _ in range(PERMUTATIONS)]
        )
        continuous_effects[column] = {
            "n": int(valid.sum()),
            "eta_squared": observed,
            "permutation_p_value": float(
                (1 + (null >= observed - 1e-12).sum()) / (PERMUTATIONS + 1)
            ),
        }

    association = {
        "n_songs": n_songs,
        "chi_square": float(chi_square),
        "degrees_of_freedom": int(degrees_freedom),
        "asymptotic_p_value": float(asymptotic_p),
        "cramers_v": cramers_v,
        "bias_corrected_cramers_v": corrected_v,
        "minimum_expected_count": float(expected.min()),
        "expected_cells_below_5": int((expected < 5).sum()),
        "permutation_p_value": (extreme + 1) / (PERMUTATIONS + 1),
        "permutations": PERMUTATIONS,
        "seed": SEED,
        "hangul_dominant_mixed_latin_dominant_other_and_no_letters_counts": {
            stratum: int((scripts["script_stratum"] == stratum).sum()) for stratum in STRATA
        },
        "continuous_share_effects": continuous_effects,
        "interpretation_caveat": (
            "These are character-script proxies, not validated language identification. "
            "Permutation inference assumes song-level exchangeability and does not model "
            "artist/version or graph dependence; association is not causation."
        ),
    }
    write_csv(pd.DataFrame([association]), "script_association.csv")
    write_json(
        {
            "selected_seed": selected_seed,
            "selected_community_count": int(np.unique(selected_labels).size),
            "selected_task_b_reported_modularity": float(run_modularity.loc[selected_seed]),
            "selected_standard_modularity_evaluation_gamma_1": partition_rows[selected_seed][
                "standard_modularity_evaluation_gamma_1"
            ],
            "selected_mean_nmi": float(mean_nmi[selected_seed]),
            "selected_mean_ari": float(mean_ari[selected_seed]),
            "community_sizes_descending": json.loads(
                partition_rows[selected_seed]["community_sizes_descending"]
            ),
            "partition_selection_method": (
                "Highest mean NMI to the other 19 seeds; ties broken by higher Task B "
                "modularity, then lower seed."
            ),
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "graph_components": nx.number_connected_components(graph),
            "embedding_shape": list(embeddings.shape),
            "script_association": association,
            "input_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    corpus_path,
                    embedding_path,
                    embedding_ids_path,
                    TASK_B / "louvain_repeated_runs.csv",
                    TASK_C / "baseline_labels.npy",
                    TASK_C / "song_ids.npy",
                    TASK_C / "graph.npz",
                )
            },
        },
        "analysis_metadata.json",
    )
    print(
        json.dumps(
            {
                "selected_seed": selected_seed,
                "communities": int(np.unique(selected_labels).size),
                "community_sizes": sorted(
                    np.bincount(selected_labels).tolist(), reverse=True
                ),
                "selected_task_b_reported_modularity": float(run_modularity.loc[selected_seed]),
                "mean_nmi": float(mean_nmi[selected_seed]),
                "mean_ari": float(mean_ari[selected_seed]),
                "script_chi_square": float(chi_square),
                "cramers_v": cramers_v,
                "bias_corrected_cramers_v": corrected_v,
                "script_permutation_p": association["permutation_p_value"],
                "script_counts": association[
                    "hangul_dominant_mixed_latin_dominant_other_and_no_letters_counts"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
