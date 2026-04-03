from __future__ import annotations

import re
from datetime import datetime, timezone

from .query_intent import QueryIntent

def _score_recency(dataset: dict) -> float:
    raw = dataset.get('last_updated', '')
    if not raw:
        return 0.8

    try:
        for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m', '%Y'):
            try:
                dt = datetime.strptime(str(raw)[:10], fmt[:len(fmt)])
                break
            except ValueError:
                continue
        else:
            return 0.8

        age_years = (datetime.now() - dt).days / 365.25
        return max(0.4, 1.0 - (age_years / 10.0))
    except Exception:
        return 0.8

_LICENSE_SCORES: dict[str, float] = {
    'public domain': 1.0,
    'cc0':           1.0,
    'cc-by':         0.9,
    'cc by':         0.9,
    'mit':           0.9,
    'apache':        0.85,
    'open':          0.8,
    'cc-by-nc':      0.6,
    'cc by-nc':      0.6,
    'research only': 0.5,
    'non-commercial':0.5,
    'proprietary':   0.2,
    'unknown':       0.8,
    '':              0.8,
}

def _score_license(dataset: dict) -> float:
    lic = str(dataset.get('license', '')).lower().strip()
    for key, score in _LICENSE_SCORES.items():
        if key and key in lic:
            return score
    return 0.8

def _score_size(dataset: dict) -> float:
    raw = str(dataset.get('size_estimate', '')).lower()
    if not raw or raw in ('unknown', 'n/a', ''):
        return 0.8

    match = re.search(r'([\d,.]+)\s*(kb|mb|gb|tb|rows?|samples?|k|m)', raw)
    if not match:
        return 0.8

    num = float(match.group(1).replace(',', ''))
    unit = match.group(2)

    mb_map = {
        'kb': num / 1024,
        'mb': num,
        'gb': num * 1024,
        'tb': num * 1024 * 1024,
        'rows': num / 10_000,
        'row':  num / 10_000,
        'samples': num / 10_000,
        'sample':  num / 10_000,
        'k':  num / 10,
        'm':  num * 100,
    }
    size_mb = mb_map.get(unit, num)

    if size_mb < 0.1:
        return 0.3
    elif size_mb < 1:
        return 0.5
    elif size_mb < 10:
        return 0.7
    elif size_mb < 500:
        return 0.9
    else:
        return 1.0

_FORMAT_SCORES: dict[str, float] = {
    'csv':     1.0,
    'parquet': 0.95,
    'json':    0.9,
    'jsonl':   0.9,
    'tsv':     0.85,
    'xlsx':    0.7,
    'hdf5':    0.75,
    'h5':      0.75,
    'npy':     0.7,
    'zip':     0.9,
    'tar':     0.8,
    'api':     0.8,
}

def _score_format(dataset: dict) -> float:
    fmt = str(dataset.get('format', '')).lower()
    for key, score in _FORMAT_SCORES.items():
        if key in fmt:
            return score
    return 0.8

def _passes_constraint(dataset: dict, key: str, value: object) -> bool:
    text = " ".join([
        str(dataset.get("name", "")),
        str(dataset.get("description", "")),
        str(dataset.get("suitability_notes", "")),
    ]).lower()

    if key == "class_balance":
        if value == "imbalanced":
            return bool(re.search(
                r"\bimbalanced?\b|\bclass[\s_-]?imbalance\b|\bskewed\b"
                r"|\brare[\s_-]?class\b|\bminority[\s_-]?class\b"
                r"|\bunequal[\s_-]?class\b",
                text
            ))
        if value == "balanced":
            return bool(re.search(r"\bbalanced\b|\bequal[\s_-]?class\b", text))

    elif key == "labeled":
        if value is True:
            return bool(re.search(
                r"\blabell?ed\b|\bannotated\b|\bground[\s_-]?truth\b", text
            ))
        if value is False:
            return bool(re.search(r"\bunlabell?ed\b|\bunsupervised\b", text))

    elif key == "modality":
        modality_patterns = {
            "time_series": r"\btime[\s_-]?series\b|\btemporal\b|\bsequential\b",
            "image":       r"\bimage[s]?\b|\bvision\b|\bphoto[s]?\b|\bvisual\b",
            "text":        r"\btext\b|\bnlp\b|\bcorpus\b|\bsentence[s]?\b",
            "audio":       r"\baudio\b|\bspeech\b|\bsound\b",
            "tabular":     r"\btabular\b|\bcsv\b|\bspreadsheet\b|\bstructured\b",
            "graph":       r"\bgraph\b|\bnetwork\b|\bnode[s]?\b|\bedge[s]?\b",
        }
        pattern = modality_patterns.get(str(value), "")
        return bool(re.search(pattern, text)) if pattern else True

    elif key == "license_type":
        lic = str(dataset.get("license", "")).lower()
        if value == "commercial_friendly":
            return not bool(re.search(r"\bnon[\s_-]?commercial\b|\bnc\b|\bresearch[\s_-]?only\b", lic))
        if value in ("open", "public_domain"):
            return bool(re.search(
                r"\bopen\b|\bcc0\b|\bpublic[\s_-]?domain\b|\bmit\b|\bapache\b|\bcc[\s_-]?by\b", lic
            ))

    elif key == "min_size":
        return _score_size(dataset) >= 0.6

    elif key == "max_size":
        return _score_size(dataset) <= 0.5

    elif key == "min_recency":
        return _score_recency(dataset) >= 0.6

    elif key == "is_benchmark":
        return bool(re.search(r"\bbenchmark\b|\bsota\b|\bleaderboard\b", text))

    elif key == "multimodal":
        return bool(re.search(r"\bmultimodal\b|\bmulti[\s_-]?modal\b", text))

    return True

def hard_filter(
    datasets: list[dict],
    intent: QueryIntent,
) -> tuple[list[dict], list[dict]]:
    if not intent.hard_constraints:
        return datasets, []

    passed, rejected = [], []
    for ds in datasets:
        if all(
            _passes_constraint(ds, key, value)
            for key, value in intent.hard_constraints.items()
        ):
            passed.append(ds)
        else:
            rejected.append(ds)

    return passed, rejected

_BASE_WEIGHTS: dict[str, float] = {
    "llm_score":           0.40,
    "semantic_similarity": 0.40,
    "recency":             0.05,
    "license_openness":    0.05,
    "size_score":          0.05,
    "format_match":        0.05,
}

def resolve_weights(
    intent: QueryIntent,
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = dict(base_weights or _BASE_WEIGHTS)

    for dim, boost in intent.weight_boosts.items():
        if dim in weights:
            weights[dim] = weights[dim] + boost

    weights = {k: max(0.0, min(1.0, v)) for k, v in weights.items()}

    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def score_results(
    datasets: list[dict],
    query: str,
    intent: QueryIntent,
    semantic_scores: dict[str, float] | None = None,
    llm_scores: dict[str, float] | None = None,
    base_weights: dict[str, float] | None = None,
) -> list[dict]:
    weights = resolve_weights(intent, base_weights)
    semantic_scores = semantic_scores or {}
    llm_scores = llm_scores or {}

    scored = []
    for ds in datasets:
        url_key = ds.get('url', '')

        dim_scores = {
            'llm_score':           llm_scores.get(url_key, 0.5),
            'semantic_similarity': semantic_scores.get(url_key, 0.5),
            'recency':             _score_recency(ds),
            'license_openness':    _score_license(ds),
            'size_score':          _score_size(ds),
            'format_match':        _score_format(ds),
        }

        relevance_score = sum(
            dim_scores[dim] * weights[dim]
            for dim in weights
        )

        scored.append({
            **ds,
            'relevance_score': round(relevance_score, 4),
            'dim_scores':      {k: round(v, 3) for k, v in dim_scores.items()},
            'active_weights':  {k: round(v, 3) for k, v in weights.items()},
            'active_constraints': intent.hard_constraints,
        })

    return sorted(scored, key=lambda x: x['relevance_score'], reverse=True)