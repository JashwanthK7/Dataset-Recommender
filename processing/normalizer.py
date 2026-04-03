def normalize_results(raw_results: list[dict]) -> list[dict]:
    normalized = []
    for item in raw_results:
        normalized.append({
            'source': str(item.get('source') or 'Unknown'),
            'name': str(item.get('name') or 'Untitled Dataset'),
            'description': str(item.get('description') or ''),
            'url': str(item.get('url') or ''),
            'license': str(item.get('license') or 'unknown'),
            'last_updated': str(item.get('last_updated') or 'unknown'),
            'format': str(item.get('format') or 'unknown'),
            'size_estimate': str(item.get('size_estimate') or 'unknown'),
        })
    return normalized