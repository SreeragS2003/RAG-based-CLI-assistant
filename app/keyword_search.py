#Count how many query words appear in the chunk text, and return the total count as a score. This can be used as a simple keyword-based relevance score to rank chunks based on how many query words they contain, which can be useful for filtering or re-ranking results from the vector store before passing them to the LLM for answer generation.
def keyword_score(query, text):
    query_words = query.lower().split()
    text_words = text.lower().split()

    score = 0
    for word in query_words:
        score += text_words.count(word)

    return score