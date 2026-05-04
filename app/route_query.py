def route_query(query: str):
    keywords = ["latest", "news", "today", "current", "recent"]

    if any(k in query.lower() for k in keywords):
        return "web"

    return "rag"