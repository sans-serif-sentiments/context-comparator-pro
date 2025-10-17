---
id: coding_explainer
difficulty: medium
goal: explain code behavior
metrics: [faithfulness, structure, conciseness]
expected_structure: headings_and_code
-----------------------------------------------

Prompt:
Explain to a junior developer how the following Python snippet implements caching and where it might fail. Use sections titled "Overview", "How It Works", and "Edge Cases", and reference the code block provided.

Code:
```python
cache = {}

def get_user(user_id, db):
    if user_id not in cache:
        cache[user_id] = db.fetch_user(user_id)
    return cache[user_id]
```

