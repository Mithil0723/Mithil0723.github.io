"""Run all 10 test cases from Claude page 06 against the local chatbot."""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

test_cases = [
    {"id": 1, "input": "hi", "expected": "Short greeting, no RAG pipeline triggered"},
    {"id": 2, "input": "hello there!", "expected": "Short greeting, no RAG pipeline triggered"},
    {"id": 3, "input": "what's the weather today?", "expected": "Polite out-of-scope redirect"},
    {"id": 4, "input": "what projects has Mithil built?", "expected": "Accurate list from context, no fabrication"},
    {"id": 5, "input": "is Mithil a Data Scientist?", "expected": "Clearly says NO - BS Senior in Data Science"},
    {"id": 6, "input": "what is Mithil's GPA?", "expected": "Fallback message - not in context"},
    {"id": 7, "input": "tell me about the RAG chatbot", "expected": "Accurate description from context only"},
    {"id": 8, "input": "does Mithil know React?", "expected": "Fallback or honest 'not mentioned'"},
    {"id": 9, "input": "thanks", "expected": "Short acknowledgment, no RAG pipeline"},
    {"id": 10, "input": "what companies has Mithil worked at?", "expected": "Fallback - no employment in context"},
]

results = []
for tc in test_cases:
    print(f"\n{'='*60}")
    print(f"Test {tc['id']}: {tc['input']}")
    print(f"Expected: {tc['expected']}")
    print(f"{'='*60}")
    try:
        r = requests.post(f"{BASE_URL}/chat", json={"message": tc["input"]}, timeout=30)
        reply = r.json().get("reply", "ERROR: no reply field")
        print(f"Actual: {reply}")
        results.append({"id": tc["id"], "input": tc["input"], "expected": tc["expected"], "actual": reply})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"id": tc["id"], "input": tc["input"], "expected": tc["expected"], "actual": f"ERROR: {e}"})

# Save results to file
with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open("test_results.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"\n{'='*60}\n")
        f.write(f"Test {r['id']}: {r['input']}\n")
        f.write(f"Expected: {r['expected']}\n")
        f.write(f"Actual: {r['actual']}\n")

print("\n\nResults saved to test_results.json and test_results.txt")
