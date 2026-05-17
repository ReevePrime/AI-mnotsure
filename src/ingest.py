import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

DOCUMENT_PATH = "data/source.txt"
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
OVERLAP_CHARS = 50
MIN_CHUNK_CHARS = 200  # paragraphs shorter than this are merged forward

load_dotenv()
client = OpenAI()

##########################################################################
#                         CHUNKING FUNCTIONS
##########################################################################


def load_document(path: str) -> str:
    """Function to load our document from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def paragraph_chunks(text: str, overlap_chars: int = OVERLAP_CHARS, min_chunk_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Split text at double new lines (\n\n), then merge short paragraphs forward
    so that small subsections (e.g. a two-line PRICING paragraph) are never
    isolated chunks with too little embedding signal."""

    raw = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Accumulate paragraphs into a buffer until it reaches min_chunk_chars
    merged = []
    buf = ""
    for para in raw:
        buf = (buf + "\n\n" + para) if buf else para
        if len(buf) >= min_chunk_chars:
            merged.append(buf)
            buf = ""
    # Flush any leftover into the last chunk rather than creating a tiny tail
    if buf:
        if merged:
            merged[-1] += "\n\n" + buf
        else:
            merged.append(buf)

    chunks = []
    for i, para in enumerate(merged):
        if i == 0:
            chunks.append(para)
        else:
            tail = chunks[-1][-overlap_chars:]
            chunks.append(tail + ' ' + para)

    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    """We batch all our chunks together and send them to the API to be embedded."""
    response = client.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    return [item.embedding for item in response.data]


##########################################################################
#                         BUILDING FAISS INDEX
##########################################################################


def build_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    """We create an IndexFlatL2 and add all embedding vectors to it."""

    # FAISS isn't LLM specific. It's a general-purpose vector search library.
    # So we need to specify how many dimensions our vectors have.
    # Our embedding model is text-embedding-3-small. It produces 1536-dimensional vectors.
    # So we tell FAISS to expect 1536-dimensional vectors.
    index = faiss.IndexFlatL2(EMBED_DIM)
    # FAISS expects a 2D numpy array of shape (n_vectors, d) (dtype must be float32 for IndexFlatL2)
    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)
    return index


def save_artifacts(index: faiss.IndexFlatL2, chunks: list[str]) -> None:
    """
    Write the FAISS index and chunk list to disk.
    Creates the data/ directory if it doesn't already exist.
    """
    Path("data").mkdir(exist_ok=True)

    # FAISS has its own binary format for storing indexes.
    # We just pass it the index object and a file path string and it writes a binary .faiss file to disk.
    faiss.write_index(index, INDEX_PATH)

    # Sanity check
    print(f"  Saved FAISS index  → {INDEX_PATH}  ({index.ntotal} vectors)")

    with open(CHUNKS_PATH, "wb") as f:
        # We use pickle to serialize the list of chunk strings
        # so we recognize which chunk corresponds to which vector in the FAISS index.
        pickle.dump(chunks, f)
    print(f"  Saved chunk list   → {CHUNKS_PATH}  ({len(chunks)} chunks)")


##########################################################################
#                         MAIN INGEST FUNCTION
##########################################################################

def main():

    text = load_document(DOCUMENT_PATH)
    print(f"  Loaded {len(text):,} characters from {DOCUMENT_PATH}")

    chunks = paragraph_chunks(text)
    print(f"  Created {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(
            f"  Chunk {i:02d}: {len(c):>4} chars — {c[:60].replace(chr(10), ' ')}...")

    embeddings = embed_batch(chunks)
    print(
        f"  Generated {len(embeddings)} embeddings, each {len(embeddings[0])} dimensions")

    index = build_index(embeddings)
    print(f"  Index contains {index.ntotal} vectors")

    save_artifacts(index, chunks)
    print("\n✓ Ingestion complete.")


if __name__ == "__main__":
    main()
