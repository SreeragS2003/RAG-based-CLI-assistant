from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# Load FlashRank model once at module level — avoid reloading on every call
# ms-marco-MiniLM-L-12-v2 is a lightweight cross-encoder, good balance of speed and quality
flashrank_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")

def rerank(query, candidates, distances):
    if not candidates:
        return []

    # ── Step 1: BM25 keyword scoring ──────────────────────────────────────────
    # Tokenize all candidate documents for BM25
    tokenized_corpus = [c["content"].lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    # Get BM25 scores for the query — replaces the manual keyword scoring
    bm25_scores = bm25.get_scores(query.lower().split())

    # Normalize BM25 scores to 0-1 range so they're comparable with semantic scores
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

    # ── Step 2: Semantic scoring ─────────────────────────────
    # Convert cosine distance to similarity score
    sem_scores = [1 / (1 + dist) for dist in distances]

    # ── Step 3: Combine BM25 + semantic scores ────────────────────────────────
    # Weighted combination — BM25 handles keyword matching, semantic handles meaning
    combined = []
    for i, c in enumerate(candidates):
        total_score = (0.5 * bm25_scores_normalized[i]) + (0.5 * sem_scores[i])
        combined.append((total_score, c))

    # Sort by combined score and take top 20 for FlashRank
    combined.sort(reverse=True, key=lambda x: x[0])
    top_candidates = [c for _, c in combined]

    # ── Step 4: FlashRank cross-encoder reranking ─────────────────────────────
    # FlashRank uses a cross-encoder that jointly processes query + document
    # Much more accurate than BM25 or cosine similarity alone, but slower
    # So we only run it on the top candidates after initial filtering
    passages = [
        {"id": i, "text": c["content"], "meta": {"source": c["source"]}}
        for i, c in enumerate(top_candidates)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    flashrank_results = flashrank_ranker.rerank(rerank_request)

    # Reconstruct candidates in original format from FlashRank results
    reranked = [
        {
            "content": r["text"],
            "source": r["meta"]["source"],
            "chunk_id": top_candidates[r["id"]]["chunk_id"]
        }
        for r in flashrank_results
    ]

    return reranked