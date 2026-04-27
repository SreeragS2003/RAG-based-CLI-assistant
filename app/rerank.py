def rerank(query, candidates, distances):
    scored = []

    q = query.lower()

    for c, dist in zip(candidates, distances):
        text = c["content"].lower()

        # 1. Semantic score (convert distance → similarity)
        sem_score = 1 / (1 + dist)

        # 2. Keyword score
        key_score = 0

        # exact phrase boost
        if q in text:
            key_score += 5

        # word overlap
        for word in q.split():
            if word in text:
                key_score += 1

        # 3. Combine scores (weighted)
        total_score = (0.7 * key_score) + (0.3 * sem_score)

        scored.append((total_score, c))

    # 4. Sort by combined score
    scored.sort(reverse=True, key=lambda x: x[0])

    return [c for _, c in scored]