import asyncio
import traceback

import gradio as gr
import re
import config
from retrievers.kaggle import KaggleRetriever
from retrievers.huggingface import HuggingFaceRetriever
from retrievers.datagov import DataGovRetriever
from processing.normalizer import normalize_results
from processing.embedder import Embedder
from processing.query_intent import parse_query_intent
from processing.scorer import hard_filter, score_results
from processing.faiss_index import FAISSIndex
from ranking.llm_ranker import LLMRanker
from ranking.result_builder import build_result_cards

embedder = Embedder(model_name=config.EMBEDDING_MODEL)
ranker = LLMRanker(
    model_id=config.LLM_MODEL_ID)

async def _fetch_all(query: str) -> list[dict]:
    secrets = config.check_secrets()

    retrievers = [
        HuggingFaceRetriever(max_results=config.RESULTS_PER_SOURCE),
        DataGovRetriever(
            max_results=config.RESULTS_PER_SOURCE,
            timeout=config.RETRIEVER_TIMEOUT_SECONDS,
        ),
    ]
    if secrets["kaggle"]:
        retrievers.append(
            KaggleRetriever(
                username=config.KAGGLE_USERNAME,
                key=config.KAGGLE_KEY,
                max_results=config.RESULTS_PER_SOURCE,
            )
        )

    tasks = [r.fetch(query) for r in retrievers]
    results_per_lane = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[dict] = []
    for lane_result in results_per_lane:
        if isinstance(lane_result, Exception):
            print(f"[retriever error] {lane_result}")
            continue
        merged.extend(lane_result)

    return merged


def run_pipeline(query: str) -> tuple[str, list[dict]]:
    query = query.strip()
    if not query:
        return "Please enter a research question.", []

    try:
        intent = parse_query_intent(query)
        print(f"[intent] {intent}")

        raw_results = asyncio.run(_fetch_all(query))
        
        source_list = [res.get("source") for res in raw_results]
        source_counts = {src: source_list.count(src) for src in set(source_list)}
        print(f"DEBUG SOURCE COUNTS: {source_counts}")

        if not raw_results:
            return "No datasets found. Try rephrasing your question.", []

        normalized = normalize_results(raw_results)

        candidates, rejected = hard_filter(normalized, intent)

        if not candidates:
            candidates = normalized
            filter_warning = (
                f" Note: no datasets matched all constraints "
                f"({intent.summary()}), showing best available results."
            )
        else:
            filter_warning = ""

        query_embedding = embedder.embed_query(query)
        dataset_embeddings = embedder.embed_datasets(candidates)

        index = FAISSIndex()
        index.build(dataset_embeddings)
        top_indices = index.search(query_embedding, k=config.FAISS_TOP_K)
        faiss_candidates = [candidates[i] for i in top_indices]

        raw_semantic_scores = {
            candidates[i]['url']: float(score)
            for i, score in zip(top_indices, index.last_scores)
        }
        
        semantic_scores = {}
        if raw_semantic_scores:
            max_score = max(raw_semantic_scores.values())
            min_score = min(raw_semantic_scores.values())
            
            for url, score in raw_semantic_scores.items():
                if max_score > min_score:
                    norm_score = 0.8 + 0.2 * ((score - min_score) / (max_score - min_score))
                else:
                    norm_score = 1.0
                semantic_scores[url] = norm_score

        scored = score_results(
            datasets=faiss_candidates,
            query=query,
            intent=intent,
            semantic_scores=semantic_scores,
        )

        seen_names = set()
        deduplicated = []
        for ds in scored:
            clean_name = re.sub(r'[^a-z0-9]', '', str(ds.get('name', '')).lower())
            if clean_name not in seen_names:
                seen_names.add(clean_name)
                deduplicated.append(ds)

        ranked = ranker.rank(
            query=query,
            candidates=deduplicated[: config.LLM_INPUT_COUNT],
            intent_context=intent.context_signals,
            active_constraints=intent.hard_constraints,
        )

        for ds in ranked:
            weights = ds['active_weights']
            ds['dim_scores']['llm_score'] = ds.get('llm_score', 0.5)
            
            final_score = sum(ds['dim_scores'][dim] * weights[dim] for dim in weights)
            ds['relevance_score'] = round(final_score, 4)

        ranked = sorted(ranked, key=lambda x: x['relevance_score'], reverse=True)

        cards = build_result_cards(ranked, top_n=config.DISPLAY_TOP_N)

        sources_used = len({c["source"] for c in normalized})
        n_filtered = len(rejected)
        filter_note = (
            f" ({n_filtered} filtered by query constraints)"
            if n_filtered and not filter_warning
            else ""
        )

        status = (
            f"Found {len(raw_results)} datasets across {sources_used} source(s). "
            f"Showing top {len(cards)} ranked by suitability."
            f"{filter_note}{filter_warning}"
        )
        return status, cards

    except Exception:
        traceback.print_exc()
        return "An error occurred. Check the logs for details.", []

def _format_cards_as_markdown(cards: list[dict]) -> str:
    if not cards:
        return ""

    lines = []
    for card in cards:
        source_badge = f"`{card['source']}`"
        score_pct = f"{card['relevance_score']:.0%}"

        lines.append(f"### {card['rank']}. {card['name']}  {source_badge}")
        lines.append(
            f"**Suitability score:** {score_pct}  |  "
            f"**Format:** {card.get('format', 'N/A')}  |  "
            f"**License:** {card.get('license', 'N/A')}  |  "
            f"**Updated:** {card.get('last_updated', 'N/A')}"
        )

        constraints = card.get("active_constraints")
        if constraints:
            c_str = ", ".join(f"`{k}={v}`" for k, v in constraints.items())
            lines.append(f"_Matched constraints: {c_str}_")

        lines.append(f"\n{card.get('suitability_notes', '')}")
        if card.get("url"):
            lines.append(f"\n[View dataset →]({card['url']})")
        lines.append("\n---")

    return "\n".join(lines)

def gradio_handler(query: str):
    status, cards = run_pipeline(query)
    return status, _format_cards_as_markdown(cards)

_secrets = config.check_secrets()
_missing = [k for k, v in _secrets.items() if not v]
if _missing:
    print(
        f"[config] Optional secrets not set: {', '.join(_missing)}. "
    )

with gr.Blocks(title="Dataset Recommender") as demo:
    gr.Markdown("## Dataset Recommender")
    gr.Markdown(
        "Describe your research question and get ranked, open dataset recommendations "
        "from Kaggle, Hugging Face and data.gov, simultaneously."
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Research question or problem description",
            placeholder="e.g. imbalanced dataset for fraud detection with labeled transactions",
            lines=3,
            scale=4,
        )
        submit_btn = gr.Button("Find datasets", variant="primary", scale=1)

    status_box = gr.Textbox(label="Status", interactive=False, lines=2)
    results_box = gr.Markdown(label="Results")

    submit_btn.click(
        fn=gradio_handler,
        inputs=[query_box],
        outputs=[status_box, results_box],
    )
    query_box.submit(
        fn=gradio_handler,
        inputs=[query_box],
        outputs=[status_box, results_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)