from rank_bm25 import BM25Okapi

def rerank(query, candidates, distances):
    if not candidates:
        return []

    # ── Step 1: BM25 keyword scoring ──────────────────────────
    tokenized_corpus = [c["content"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = bm25.get_scores(query.lower().split())

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

    # ── Step 2: Semantic scoring ─────────────────────────────
    sem_scores = [1 / (1 + dist) for dist in distances]

    # ── Step 3: Combine scores ────────────────────────────────
    combined = []
    for i, c in enumerate(candidates):
        total_score = (0.5 * bm25_scores_normalized[i]) + (0.5 * sem_scores[i])
        combined.append((total_score, c))

    # Sort by combined score
    combined.sort(reverse=True, key=lambda x: x[0])

    # Return sorted results
    reranked = [c for _, c in combined]

    return reranked