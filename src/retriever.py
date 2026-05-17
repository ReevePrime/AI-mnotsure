import pickle
import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
INDEX_PATH = BASE_DIR / "data" / "index.faiss"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.pkl"

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 3       # number of chunks to retrieve per query
# Rule of thumb: 3 is a good number to balance relevance and diversity
# More might make it noisy, less might miss relevant info.

MAX_DISTANCE = 1    # we filter out chunks that are less similar than this threshold

load_dotenv(override=True)
client = OpenAI()


##########################################################################
#                         LOADING FAISS INDEX
##########################################################################

def load_index() -> faiss.IndexFlatL2:
    """Load the FAISS index"""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. "
            "Please make sure to run src/ingest.py first."
        )
    return faiss.read_index(str(INDEX_PATH))


def load_chunks() -> list[str]:
    """Load the chunk list"""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"No chunk list found at {CHUNKS_PATH}. "
            "Please make sure to run src/ingest.py first."
        )
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


##########################################################################
#                         QUERYING FUNCTIONS
##########################################################################

def embed_query(query: str) -> np.ndarray:
    """
    Like for embed_batch. We need to use the same model.
    FAISS expectsa (1, 1536) numpy array for search.
    """
    response = client.embeddings.create(
        input=query,
        model=EMBED_MODEL
    )
    # list of 1536 floats since we use text-embedding-3-small
    vector = response.data[0].embedding
    # putting the vector in a list to make it 2D (1, 1536)
    return np.array([vector], dtype=np.float32)


def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    k: int = TOP_K,
    max_distance: float = MAX_DISTANCE
) -> list[dict]:
    """
    Embed the query, search the index, and return the top matching chunks.

    Returns a list of dicts containing 'text', 'index' and'distance'
    respectively the chunk text, its index in the original list, and its
    L2 distance from the query
    The lower the L2 distance, the more similar the chunk is to the query.

    Chunks with distance >= max_distance are filtered out to only return relevant results.
    """

    query_vector = embed_query(query)
    # index.search searches multiple queries at once, but we only have one query, so we get (1, k) arrays
    distances, indices = index.search(query_vector, k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        # if the index has fewer chunks than k, FAISS will pad the results with "-1" in the indices array.
        # we need to filter those out since they don't correspond to any chunk.
        # This is common practice when using FAISS (review later)
        if idx == -1:
            continue

        if dist >= max_distance:
            continue                      # not similar enough to be useful
        results.append({
            "text": chunks[int(idx)],
            "index": int(idx),
            "distance": float(dist)
        })

    return results


##########################################################################
#                         STANDALONE TESTING
##########################################################################


def main():
    print("Loading index and chunks...")
    index = load_index()
    chunks = load_chunks()
    print(f"  Index: {index.ntotal} vectors")
    print(f"  Chunks: {len(chunks)} strings\n")

    # Test queries to probe specific parts of the document.
    # Good retrieval means the right section comes back as the top result.
    test_queries = [
        "When was Moorepay founded?",
        "Who owns Moorepay?",
        "What accreditations does Moorepay have?",
        "How many clients does Moorepay support?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        results = retrieve(query, index, chunks)

        if not results:
            print("  ✗ No results above similarity threshold\n")
            continue

        for i, r in enumerate(results):
            print(f"  Result {i + 1} (distance: {r['distance']:.3f})")
            # Print first 120 chars of the chunk to judge relevance
            preview = r['text'][:120].replace('\n', ' ')
            print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()

# Expected output (approximate):

# Query: When was Moorepay founded?
#   Result 1 (distance: 0.312)
#     Moorepay is a UK-based payroll and HR software and services company, widely
#     regarded as a market leader. It was founded in 1966...
#   Result 2 (distance: 0.489)
#     Founded in 1966, Moorepay has over 50 years of experience...
#   Result 3 (distance: 0.701)
#     ...

# Results 1 and 2 are relevant and should be returned,
# while result 3 is less relevant and may be filtered out depending on the MAX_DISTANCE threshold.

# Rule of thumb for distances for text-embedding-3-small:
# - Below 0.5: strong semantic match
# - 0.5 – 1.0: related but not tightly matched
# - Above 1.5: probably not very relevant
# Result 3 with distance 0.701 is borderline and may or may not be returned
# if we set the MAX_DISTANCE threshold to 0.5.


# Note of things to check for:
# - If a clearly better chunk than 1 is coming back as Result 2 or 3,
#   the chunking may be splitting that content across boundaries.
# - If everything is coming back above 1.0 for questions clearly answered in the document,
#   the overlap may be too aggressive or the chunks too fragmented.
# - If a question without answers returns zero results, the threshold filter works correctly.
