import asyncio

#Safe LLM Call Wrapper function to handle transient errors and retries when calling the LLM, which can be useful for improving robustness and reliability of the system when dealing with network issues, rate limits, or other temporary problems that may arise when interacting with the LLM API.
async def safe_llm_call(fn, retries=3, delay=1):
    for attempt in range(retries): #Retry mechanism
        try:
            return await fn()
        except Exception as e:
            print(f"[LLM ERROR] Attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                raise e
            await asyncio.sleep(delay)

    raise Exception("LLM failed after retries")