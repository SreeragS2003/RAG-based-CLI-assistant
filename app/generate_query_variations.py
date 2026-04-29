import asyncio

def expand_queries(query):
    return list(set([
            query,
            f"{query} explanation",
            f"{query} details",
            f"what is {query}",
        ]))

async def parallel_search(store, queries):
    tasks = [
        asyncio.to_thread(store.hybrid_search, q) #Internally uses hybrid_search which is thread safe, so we can run in parallel using asyncio.to_thread to avoid blocking the event loop while waiting for search results. This allows us to efficiently handle multiple queries at once and combine results later.
        for q in queries
    ]

    results = await asyncio.gather(*tasks) #Wait for all search tasks to complete and gather their results. Each task will return a list of search results corresponding to its query variation, and we will combine these results into a single list for further processing (deduplication and ranking).

    # We flatten the results as each search returns a list of results, and we want to combine them into a single list for deduplication and ranking later on.
    combined = []
    for r in results:
        combined.extend(r)

    return combined