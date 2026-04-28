import json

def parse_json(response):
    response = response.strip()

    # remove markdown fences
    if response.startswith("```"):
        response = response.strip("`").strip()
        if response.startswith("json"):
            response = response[4:].strip()

    return json.loads(response)