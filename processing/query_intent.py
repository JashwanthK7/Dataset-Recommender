from __future__ import annotations
import re

_CONSTRAINT_SIGNALS: list[tuple[str, str, object]] = [
    (r"\bimbalanced?\b|\bclass[\s_-]?imbalance\b|\bskewed\s+class",
     "class_balance", "imbalanced"),
    (r"\bbalanced\s+class|\bequal\s+class|\bbalanced\s+dataset",
     "class_balance", "balanced"),
    (r"\bunlabell?ed\b|\bno\s+labels?\b|\bunsupervised\b",
     "labeled", False),
    (r"\blabell?ed\b|\bannotated\b|\bground[\s_-]?truth\b|\bsupervised\b",
     "labeled", True),
    (r"\btime[\s_-]?series\b|\btemporal\b|\bsequential\s+data\b",
     "modality", "time_series"),
    (r"\bimage[s]?\b|\bvision\b|\bphoto[s]?\b|\bpicture[s]?\b|\bvisual\b",
     "modality", "image"),
    (r"\btext\b|\bnlp\b|\bnatural\s+language\b|\bcorpus\b|\bsentence[s]?\b",
     "modality", "text"),
    (r"\baudio\b|\bspeech\b|\bsound\b|\bacoustic\b",
     "modality", "audio"),
    (r"\btabular\b|\bspreadsheet\b|\bcsv\b|\bstructured\s+data\b",
     "modality", "tabular"),
    (r"\bgraph\b|\bnetwork\s+data\b|\bnode[s]?\b|\bedge[s]?\b",
     "modality", "graph"),
    (r"\bcommercial(?:ly)?\b|\bproprietary\b|\bfor[\s_-]?profit\b",
     "license_type", "commercial_friendly"),
    (r"\bopen[\s_-]?source\b|\bfree[\s_-]?to[\s_-]?use\b|\bcc[\s_-]?by\b",
     "license_type", "open"),
    (r"\bpublic[\s_-]?domain\b|\bcc0\b|\bgovernment\s+data\b",
     "license_type", "public_domain"),
    (r"\blarge[\s_-]?scale\b|\bbig\s+dataset\b|\bmillion[s]?\s+(?:rows?|samples?|records?)",
     "min_size", "large"),
    (r"\bsmall\s+dataset\b|\bfew[\s_-]?shot\b|\blimited\s+data\b",
     "max_size", "small"),
    (r"\brecent\b|\blatest\b|\bup[\s_-]?to[\s_-]?date\b|\b202[0-9]\b",
     "min_recency", "recent"),
    (r"\bbenchmark\b|\bstate[\s_-]?of[\s_-]?the[\s_-]?art\b|\bsota\b",
     "is_benchmark", True),
    (r"\bmultimodal\b|\bmulti[\s_-]?modal\b",
     "multimodal", True),
]

_BOOST_SIGNALS: list[tuple[str, str, float]] = [
    (r"\brecent\b|\blatest\b|\b202[0-9]\b|\bcurrent\b|\bup[\s_-]?to[\s_-]?date\b",
     "recency", +0.20),
    (r"\bhistorical\b|\barchive[d]?\b|\blegacy\b|\bold\s+data\b",
     "recency", -0.15),
    (r"\bcommercial\b|\bfor[\s_-]?profit\b",
     "license_openness", +0.20),
    (r"\bopen[\s_-]?source\b|\bfree\b|\bcc0\b|\bpublic[\s_-]?domain\b",
     "license_openness", +0.15),
    (r"\blarge[\s_-]?scale\b|\bbig\s+dataset\b|\bmillion[s]?\s+(?:rows?|samples?)",
     "size_score", +0.15),
    (r"\bsmall\b|\bfew[\s_-]?shot\b|\blimited\b",
     "size_score", -0.10),
    (r"\bcsv\b|\bspreadsheet\b|\btabular\b",
     "format_match", +0.10),
    (r"\bjson\b|\bparquet\b|\bapi\b",
     "format_match", +0.05),
    (r"\bimbalanced?\b|\bfew[\s_-]?shot\b|\banomal[yi]\b|\bout[\s_-]?of[\s_-]?distribution\b",
     "semantic_similarity", +0.10),
]

_DOMAIN_SIGNALS: list[tuple[str, str]] = [
    (r"\bclinic\w*\b|\bmedical\b|\bhospital\b|\bpatient\b|\bhealthcare\b|\bdiagnos\w*\b", "healthcare"),
    (r"\bfinance\b|\bstock\b|\bmarket\b|\btrading\b|\beconom\w*\b|\bbanking\b", "finance"),
    (r"\bclimate\b|\bweather\b|\benviron\w*\b|\bcarbon\b|\bsatellite\b", "climate_environment"),
    (r"\beducation\b|\bstudent\b|\blearning\b|\bacadem\w*\b|\bschool\b", "education"),
    (r"\bcrime\b|\bpolice\b|\blaw\s+enforce\w*\b|\bsocial\s+justice\b", "public_safety"),
    (r"\bagriculture\b|\bcrop\b|\bfarm\w*\b|\bsoil\b|\byield\b", "agriculture"),
    (r"\btransport\w*\b|\btraffic\b|\bdriving\b|\bautonomous\b|\bvehicle\b", "transportation"),
    (r"\bnlp\b|\btext\b|\blanguage\s+model\b|\bsentiment\b|\btranslation\b", "nlp"),
    (r"\bvision\b|\bimage\b|\bobject\s+detect\w*\b|\bsegment\w*\b", "computer_vision"),
]

_TASK_SIGNALS: list[tuple[str, str]] = [
    (r"\bclassif\w*\b|\bcategor\w*\b|\blabel\w*\b", "classification"),
    (r"\bregress\w*\b|\bpredict\w*\b|\bforecast\w*\b", "regression_forecasting"),
    (r"\bcluster\w*\b|\bunsupervised\b|\bsegment\w*\b", "clustering"),
    (r"\banomaly\b|\boutlier\b|\bfraud\b|\bdetect\w*\b", "anomaly_detection"),
    (r"\bgenerat\w*\b|\bsynthes\w*\b|\bdiffusion\b|\bgans?\b", "generative"),
    (r"\brecommend\w*\b|\bcollaborative\s+filter\w*\b", "recommendation"),
    (r"\bqa\b|\bquestion\s+answer\w*\b|\bcomprehension\b", "question_answering"),
]

class QueryIntent:
    def __init__(
        self,
        hard_constraints: dict,
        weight_boosts: dict[str, float],
        context_signals: dict[str, str],
        raw_query: str,
    ):
        self.hard_constraints = hard_constraints
        self.weight_boosts = weight_boosts
        self.context_signals = context_signals
        self.raw_query = raw_query

    def has_constraints(self) -> bool:
        return bool(self.hard_constraints)

    def summary(self) -> str:
        parts = []
        if self.hard_constraints:
            c = ", ".join(f"{k}={v}" for k, v in self.hard_constraints.items())
            parts.append(f"filters: {c}")
        if self.context_signals.get("domain"):
            parts.append(f"domain: {self.context_signals['domain']}")
        if self.context_signals.get("task"):
            parts.append(f"task: {self.context_signals['task']}")
        return " · ".join(parts) if parts else "general query"

    def __repr__(self) -> str:
        return (
            f"QueryIntent(constraints={self.hard_constraints}, "
            f"boosts={self.weight_boosts}, context={self.context_signals})"
        )

def parse_query_intent(query: str) -> QueryIntent:
    q = query.lower()

    hard_constraints: dict = {}
    for pattern, key, value in _CONSTRAINT_SIGNALS:
        if key not in hard_constraints and re.search(pattern, q):
            hard_constraints[key] = value

    weight_boosts: dict[str, float] = {}
    for pattern, dimension, boost in _BOOST_SIGNALS:
        if re.search(pattern, q):
            weight_boosts[dimension] = weight_boosts.get(dimension, 0.0) + boost

    weight_boosts = {k: max(-0.4, min(0.4, v)) for k, v in weight_boosts.items()}

    context: dict[str, str] = {}

    for pattern, domain in _DOMAIN_SIGNALS:
        if re.search(pattern, q):
            context["domain"] = domain
            break

    for pattern, task in _TASK_SIGNALS:
        if re.search(pattern, q):
            context["task"] = task
            break

    if re.search(r"\b202[0-9]\b", q):
        context["era"] = "2020s"
    elif re.search(r"\b201[0-9]\b", q):
        context["era"] = "2010s"
    elif re.search(r"\bhistorical\b|\barchive\b|\blegacy\b", q):
        context["era"] = "historical"

    return QueryIntent(
        hard_constraints=hard_constraints,
        weight_boosts=weight_boosts,
        context_signals=context,
        raw_query=query,
    )