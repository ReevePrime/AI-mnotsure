
import sys
sys.path.insert(0, "src")

from retriever import load_index, load_chunks, retrieve
index = load_index()
chunks = load_chunks()

results = retrieve(
    "What does Moorepay's starting price look like?", index, chunks, max_distance=9999)
for r in results:
    print(f"Distance: {r['distance']:.3f}")
    print(r['text'][:200])
    print("---")
