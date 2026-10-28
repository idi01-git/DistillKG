import json

log_path = r"C:\Users\shiva\.gemini\antigravity\brain\27c29fbf-2ead-43a2-975b-ae4b6d60695a\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if '"type":"USER_INPUT"' in line:
            data = json.loads(line)
            content = data.get("content", "")
            print(f"Step {i}: {content}\n---")

