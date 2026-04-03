def build_result_cards(ranked_datasets: list[dict], top_n: int = 5) -> list[dict]:
    cards = []
    for i, ds in enumerate(ranked_datasets[:top_n]):
        card = {
            'rank': i + 1,
            'name': ds.get('name') or 'Unknown',
            'source': ds.get('source') or 'Unknown',
            'relevance_score': ds.get('relevance_score', 0),
            'format': ds.get('format') or 'N/A',
            'license': ds.get('license') or 'N/A',
            'last_updated': ds.get('last_updated') or 'N/A',
            'suitability_notes': ds.get('suitability_notes') or '',
            'url': ds.get('url') or '',
            'active_constraints': ds.get('active_constraints') or {}
        }
        cards.append(card)
    return cards