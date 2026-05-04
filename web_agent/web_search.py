from ddgs import DDGS

def web_search(query: str):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))

    formatted = []
    for r in results:
        formatted.append(
            f"{r.get('title')}\n{r.get('body')}\n{r.get('href')}"
        )

    return "\n\n".join(formatted)