import json
import math
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from .reward import extract_presented_answer, normalize_generated_completion

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _extract_final_answer_text(completion: str) -> str:
    text = str(completion or "")
    normalized = normalize_generated_completion(text, answer_prefix="")
    presented = extract_presented_answer(text)
    if presented:
        return presented.strip()
    canonical = normalized.get("completion_canonical")
    if canonical:
        return str(canonical).strip()

    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1).strip()

    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _safe_mean(values)
    var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
    return float(math.sqrt(max(var, 0.0)))


def _pairwise_cosine_distance_mean(emb: np.ndarray) -> float:
    n = emb.shape[0]
    if n <= 1:
        return 0.0
    sim = emb @ emb.T
    iu = np.triu_indices(n, k=1)
    dist = 1.0 - sim[iu]
    return float(np.mean(dist))


def _dispersion_to_centroid(emb: np.ndarray) -> np.ndarray:
    if emb.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    centroid = emb.mean(axis=0, keepdims=True)
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True).clip(min=1e-12)
    centroid = centroid / centroid_norm
    sim = np.sum(emb * centroid, axis=1)
    return (1.0 - sim).astype(np.float32)


def _entropy(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts.astype(np.float64) / float(total)
    probs = probs[probs > 0.0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _normalized_entropy(counts: np.ndarray) -> float:
    if counts.size <= 1:
        return 0.0
    ent = _entropy(counts)
    denom = math.log(counts.size)
    if denom <= 0:
        return 0.0
    return float(ent / denom)


def _permutation_test_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_iters: int,
    seed: int,
) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return {
            "observed_diff": 0.0,
            "p_value_two_sided": 1.0,
            "n_a": int(a.size),
            "n_b": int(b.size),
        }

    observed = float(np.mean(a) - np.mean(b))
    joined = np.concatenate([a, b], axis=0)
    n_a = int(a.size)
    rng = np.random.default_rng(seed)

    hits = 0
    for _ in range(max(1, int(n_iters))):
        perm = rng.permutation(joined)
        pa = perm[:n_a]
        pb = perm[n_a:]
        d = float(np.mean(pa) - np.mean(pb))
        if abs(d) >= abs(observed):
            hits += 1

    p_val = float((hits + 1) / (max(1, int(n_iters)) + 1))
    return {
        "observed_diff": observed,
        "p_value_two_sided": p_val,
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def _embed_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), max(1, int(batch_size))):
            batch = [str(t or "") for t in texts[start:start + batch_size]]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max(8, int(max_length)),
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc, return_dict=True)

            pooled = None
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hs = outputs.last_hidden_state
                mask = enc.get("attention_mask", None)
                if mask is None:
                    pooled = hs.mean(dim=1)
                else:
                    mask = mask.float().unsqueeze(-1)
                    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            elif isinstance(outputs, tuple) and outputs and torch.is_tensor(outputs[0]):
                hs = outputs[0]
                pooled = hs.mean(dim=1)

            if pooled is None:
                raise RuntimeError(f"Could not extract embeddings from model output for {model_name}.")

            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype(np.float32))

    emb = np.concatenate(vectors, axis=0)
    if emb.ndim != 2:
        emb = emb.reshape(len(texts), -1)
    return emb


def _cluster_embeddings(emb: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, float]:
    n = emb.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=np.int64), 0.0

    k = max(2, min(int(k), n))
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception:
        return np.zeros((n,), dtype=np.int64), 0.0

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(emb)
    sil = 0.0
    if len(set(labels.tolist())) > 1 and n > k:
        try:
            sil = float(silhouette_score(emb, labels, metric="cosine"))
        except Exception:
            sil = 0.0
    return labels.astype(np.int64), sil


def _plot_embedding_scatter(
    coords: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: str,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)

    label_order = ["Base", "Explorer", "Actual"]
    color_map = {"Base": "#4C78A8", "Explorer": "#F58518", "Actual": "#54A24B"}

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for name in label_order:
        idx = [i for i, l in enumerate(labels) if l == name]
        if not idx:
            continue
        pts = coords[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.7, c=color_map.get(name, "#666666"), label=name)
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_diversity_bars(summary: Dict[str, Dict[str, float]], path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)

    names = ["Base", "Explorer", "Actual"]
    pairwise = [summary.get(n, {}).get("pairwise_cosine_distance_mean", 0.0) for n in names]
    diversity = [summary.get(n, {}).get("answer_diversity_ratio", 0.0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    axes[0].bar(names, pairwise, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.9)
    axes[0].set_title("Embedding dispersion")
    axes[0].set_ylabel("Mean pairwise cosine distance")
    axes[1].bar(names, diversity, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.9)
    axes[1].set_title("Answer diversity")
    axes[1].set_ylabel("Unique final answers / total")
    axes[1].set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_cluster_entropy(summary: Dict[str, Dict[str, float]], path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    os.makedirs(os.path.dirname(path), exist_ok=True)

    names = ["Base", "Explorer", "Actual"]
    vals = [summary.get(n, {}).get("cluster_entropy_norm", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(names, vals, color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.9)
    ax.set_title("Cluster occupancy entropy")
    ax.set_ylabel("Normalized entropy")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def run_embedding_novelty_analysis(
    base_samples: Sequence[Dict[str, object]],
    explorer_samples: Sequence[Dict[str, object]],
    actual_samples: Sequence[Dict[str, object]],
    cfg,
    plots_dir: str,
    analysis_dir: str,
    use_wandb: bool = False,
    wandb_run=None,
) -> Dict[str, object]:
    max_texts = max(1, int(cfg.embedding_max_texts))
    random.seed(int(cfg.seed))

    by_model_samples = {
        "Base": list(base_samples)[:max_texts],
        "Explorer": list(explorer_samples)[:max_texts],
        "Actual": list(actual_samples)[:max_texts],
    }

    rows = []
    for model_name, samples in by_model_samples.items():
        for item in samples:
            text = str(item.get("completion", "") or "")
            if not text.strip():
                continue
            rows.append(
                {
                    "model": model_name,
                    "completion": text,
                    "topic": str(item.get("topic", "unknown")),
                    "is_correct": bool(item.get("is_correct", False)),
                    "is_ood": bool(item.get("is_ood", False)),
                    "novelty_rnd": float(item.get("novelty", 0.0)),
                }
            )

    if not rows:
        return {"status": "no_samples"}

    texts = [r["completion"] for r in rows]
    emb = _embed_texts(
        texts,
        model_name=cfg.embedding_model_name,
        batch_size=cfg.embedding_batch_size,
        max_length=cfg.embedding_max_length,
    )

    model_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        model_to_idx[str(r["model"])].append(i)

    labels, sil = _cluster_embeddings(emb, k=cfg.embedding_cluster_k, seed=cfg.seed)
    k_eff = int(max(1, min(cfg.embedding_cluster_k, len(rows))))

    summary = {}
    per_model_dispersion: Dict[str, np.ndarray] = {}
    for model_name in ("Base", "Explorer", "Actual"):
        idx = model_to_idx.get(model_name, [])
        sub = emb[idx] if idx else np.zeros((0, emb.shape[1]), dtype=np.float32)
        disp = _dispersion_to_centroid(sub)
        per_model_dispersion[model_name] = disp

        answers = [_extract_final_answer_text(rows[i]["completion"]) for i in idx]
        answer_unique = len(set(a for a in answers if a))
        answer_total = max(1, len([a for a in answers if a]))
        answer_div = float(answer_unique / answer_total) if answers else 0.0

        if idx:
            model_labels = labels[np.array(idx, dtype=np.int64)]
            counts = np.bincount(model_labels, minlength=k_eff)
            ent_norm = _normalized_entropy(counts)
        else:
            ent_norm = 0.0

        summary[model_name] = {
            "n_samples": len(idx),
            "pairwise_cosine_distance_mean": _pairwise_cosine_distance_mean(sub),
            "dispersion_mean": _safe_mean(disp.tolist()),
            "dispersion_std": _safe_std(disp.tolist()),
            "answer_diversity_ratio": answer_div,
            "cluster_entropy_norm": float(ent_norm),
            "rnd_novelty_mean": _safe_mean([float(rows[i]["novelty_rnd"]) for i in idx]),
        }

    ptests = {
        "dispersion_explorer_vs_base": _permutation_test_mean_diff(
            per_model_dispersion.get("Explorer", np.array([])),
            per_model_dispersion.get("Base", np.array([])),
            n_iters=cfg.embedding_permutation_iters,
            seed=cfg.seed + 11,
        ),
        "dispersion_actual_vs_base": _permutation_test_mean_diff(
            per_model_dispersion.get("Actual", np.array([])),
            per_model_dispersion.get("Base", np.array([])),
            n_iters=cfg.embedding_permutation_iters,
            seed=cfg.seed + 17,
        ),
        "dispersion_actual_vs_explorer": _permutation_test_mean_diff(
            per_model_dispersion.get("Actual", np.array([])),
            per_model_dispersion.get("Explorer", np.array([])),
            n_iters=cfg.embedding_permutation_iters,
            seed=cfg.seed + 23,
        ),
    }

    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    payload = {
        "embedding_model_name": cfg.embedding_model_name,
        "embedding_dim": int(emb.shape[1]),
        "num_points": int(emb.shape[0]),
        "cluster_k": int(k_eff),
        "silhouette_cosine": float(sil),
        "summary_by_model": summary,
        "permutation_tests": ptests,
    }

    summary_path = os.path.join(analysis_dir, "embedding_novelty_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    ptests_path = os.path.join(analysis_dir, "permutation_tests.json")
    with open(ptests_path, "w", encoding="utf-8") as f:
        json.dump(ptests, f, indent=2)

    labels_text = [str(r["model"]) for r in rows]
    plot_paths = {}

    if cfg.embedding_run_tsne and len(rows) >= 3:
        try:
            from sklearn.manifold import TSNE

            perplexity = max(5, min(30, (len(rows) - 1) // 3))
            tsne = TSNE(
                n_components=2,
                init="pca",
                random_state=cfg.seed,
                perplexity=perplexity,
                learning_rate="auto",
            )
            coords_tsne = tsne.fit_transform(emb)
            tsne_path = os.path.join(plots_dir, "embedding_tsne.png")
            if _plot_embedding_scatter(coords_tsne, labels_text, "Embedding t-SNE (Base/Explorer/Actual)", tsne_path):
                plot_paths["embedding_tsne"] = tsne_path
        except Exception as e:
            print(f"[WARN] t-SNE plot skipped: {e}")

    if cfg.embedding_run_umap and len(rows) >= 3:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=cfg.seed)
            coords_umap = reducer.fit_transform(emb)
            umap_path = os.path.join(plots_dir, "embedding_umap.png")
            if _plot_embedding_scatter(coords_umap, labels_text, "Embedding UMAP (Base/Explorer/Actual)", umap_path):
                plot_paths["embedding_umap"] = umap_path
        except Exception as e:
            print(f"[WARN] UMAP plot skipped: {e}")

    diversity_path = os.path.join(plots_dir, "embedding_diversity_metrics.png")
    if _plot_diversity_bars(summary, diversity_path):
        plot_paths["embedding_diversity_metrics"] = diversity_path

    entropy_path = os.path.join(plots_dir, "embedding_cluster_entropy.png")
    if _plot_cluster_entropy(summary, entropy_path):
        plot_paths["embedding_cluster_entropy"] = entropy_path

    if use_wandb and wandb_run is not None and wandb is not None:
        wandb_payload = {
            "embedding/num_points": int(emb.shape[0]),
            "embedding/embedding_dim": int(emb.shape[1]),
            "embedding/silhouette_cosine": float(sil),
            "embedding/base_dispersion_mean": float(summary["Base"]["dispersion_mean"]),
            "embedding/explorer_dispersion_mean": float(summary["Explorer"]["dispersion_mean"]),
            "embedding/actual_dispersion_mean": float(summary["Actual"]["dispersion_mean"]),
            "embedding/base_answer_diversity": float(summary["Base"]["answer_diversity_ratio"]),
            "embedding/explorer_answer_diversity": float(summary["Explorer"]["answer_diversity_ratio"]),
            "embedding/actual_answer_diversity": float(summary["Actual"]["answer_diversity_ratio"]),
            "embedding/p_dispersion_explorer_vs_base": float(ptests["dispersion_explorer_vs_base"]["p_value_two_sided"]),
            "embedding/p_dispersion_actual_vs_base": float(ptests["dispersion_actual_vs_base"]["p_value_two_sided"]),
            "embedding/p_dispersion_actual_vs_explorer": float(ptests["dispersion_actual_vs_explorer"]["p_value_two_sided"]),
        }
        wandb_run.log(wandb_payload)

        for model_name in ("Base", "Explorer", "Actual"):
            arr = per_model_dispersion.get(model_name, np.array([]))
            if arr.size:
                wandb_run.log({f"embedding/{model_name.lower()}_dispersion_hist": wandb.Histogram(arr.tolist())})

        for key, path in plot_paths.items():
            wandb_run.log({f"plots/{key}": wandb.Image(path)})

    payload_with_paths = dict(payload)
    payload_with_paths["plot_paths"] = plot_paths
    payload_with_paths["summary_path"] = summary_path
    payload_with_paths["ptests_path"] = ptests_path
    return payload_with_paths
