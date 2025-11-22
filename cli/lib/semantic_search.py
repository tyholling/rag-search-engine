#!/usr/bin/env python3

import json
import os
import re

import numpy
import sentence_transformers

class SemanticSearch:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = sentence_transformers.SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if len(text) == 0 or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")

        return self.model.encode(sentences=text)

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents

        doc_strings = []
        for document in documents:
            self.document_map[document['id']] = document
            doc_strings.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(doc_strings)

        numpy.save(file='cache/movie_embeddings.npy', arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        movies = []
        for document in documents:
            self.document_map[document['id']] = document
            movies.append(f"{document['title']}: {document['description']}")

        if os.path.exists('cache/movie_embeddings.npy'):
            with open('cache/movie_embeddings.npy', 'rb') as f:
                self.embeddings = numpy.load(f)

        if self.embeddings is not None and len(self.embeddings) == len(documents):
            return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embed = self.generate_embedding(query)
        scored_docs = []
        for i in range(len(self.embeddings)):
            doc_embed = self.embeddings[i]
            score = cosine_similarity(query_embed, doc_embed)
            document = self.document_map[i + 1]
            scored_docs.append((score, document))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, document in scored_docs[:limit]:
            results.append((score, document['title'], document['description']))
        return results

# end class SemanticSearch

def verify_model():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()

    movies = []
    if os.path.exists('data/movies.json'):
        with open('data/movies.json', 'r') as f:
            movies = json.load(f)["movies"]

    embeddings = search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = numpy.dot(vec1, vec2)
    norm1 = numpy.linalg.norm(vec1)
    norm2 = numpy.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int):
    search = SemanticSearch()

    movies = []
    if os.path.exists('data/movies.json'):
        with open('data/movies.json', 'r') as f:
            movies = json.load(f)["movies"]

    embeddings = search.load_or_create_embeddings(movies)
    results = search.search(query, limit)
    for i in range(len(results)):
        score, title, description = results[i]
        print(f"{i+1}. {title} (score: {score:.4f})\n{description}\n")

def chunk_command(query: str, chunk_size: int, overlap: int):
    print(f"Chunking {len(query)} characters")
    words = query.split()
    lines = []
    while words:
        if lines and len(words) <= overlap:
            break
        chunk = words[:chunk_size]
        words = words[chunk_size - overlap:]
        lines.append(" ".join(chunk))
    for i, line in enumerate(lines):
        print(f"{i+1}. {line}")

def semantic_chunk_command(query: str, max_chunk_size: int, overlap: int):
    print(f"Semantically chunking {len(query)} characters")
    chunks = semantic_chunks(query, max_chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def semantic_chunks(query: str, max_chunk_size: int, overlap: int):
    query = query.strip()
    if query == "":
        return []
    lines = re.split(r"(?<=[.!?])\s+", query)
    if len(lines) and not lines[0].endswith(('.', '!', '?')):
        return list(lines[0])

    chunks: list[str] = []
    while lines:
        if chunks and len(lines) <= overlap:
            break
        next = lines[:max_chunk_size]
        lines = lines[max_chunk_size - overlap:]
        next_line = " ".join(next).strip()
        if next_line != "":
            chunks.append(next_line)
    return chunks

def embed_chunks_command():
    search = ChunkedSemanticSearch()

    movies = []
    if os.path.exists('data/movies.json'):
        with open('data/movies.json', 'r') as f:
            movies = json.load(f)["movies"]

    chunk_embeddings = search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self):
        super().__init__()
        self.chunk_embeddings = []
        self.chunk_metadata = []

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        chunks: list[str] = []
        metadata: list[dict] = []

        for document in documents:
            if not document['description']:
                continue
            doc_chunks = semantic_chunks(document['description'], max_chunk_size=4, overlap=1)
            chunks.extend(doc_chunks)
            for i in range(len(doc_chunks)):
                metadata.append({
                    "movie_idx": document['id'],
                    "chunk_idx": i,
                    "total_chunks": len(doc_chunks),
                })

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        with open('cache/chunk_embeddings.npy', 'wb') as f:
            numpy.save(f, self.chunk_embeddings)
        with open('cache/chunk_metadata.json', 'w') as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document

        if (os.path.exists('cache/chunk_embeddings.npy')
            and os.path.exists('cache/chunk_metadata.json')):
            with open('cache/chunk_embeddings.npy', 'rb') as f:
                self.chunk_embeddings = numpy.load(f)
            with open('cache/chunk_metadata.json', 'r') as f:
                self.chunk_metadata = json.load(f)['chunks']
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        movies = []
        if os.path.exists('data/movies.json'):
            with open('data/movies.json', 'r') as f:
                movies = json.load(f)["movies"]
        self.load_or_create_chunk_embeddings(movies)

        query_embed = self.generate_embedding(query)

        chunk_scores: list[dict] = []
        for i, chunk_embed in enumerate(self.chunk_embeddings):
            metadata = self.chunk_metadata[i]
            score = cosine_similarity(query_embed, chunk_embed)
            chunk_scores.append({
                "chunk_idx": metadata['chunk_idx'],
                "movie_idx": metadata['movie_idx'],
                "score": score,
            })

        movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx: int = chunk_score['movie_idx']
            score: float = chunk_score['score']
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score

        sorted_scores = sorted(movie_scores.items(), key=lambda kv: kv[1], reverse=True)
        results: list[dict] = []
        for movie_score in sorted_scores[:limit]:
            doc_id = movie_score[0]
            document = self.document_map[doc_id]
            results.append({
                "id": doc_id,
                "title": document['title'],
                "description": document['description'][:100],
                "score": movie_score[1],
                "metadata": self.chunk_metadata[doc_id],
            })

        return results

# end class ChunkedSemanticSearch

def search_chunked_command(query: str, limit: int):
    search = ChunkedSemanticSearch()

    movies = load_movies()
    search.load_or_create_chunk_embeddings(movies)

    results: list[dict] = search.search_chunks(query, limit)
    for i, result in enumerate(results):
        title = result['title']
        score = result['score']
        description = result['description']
        print(f"\n{i+1}. {title} (score: {score:.4f})")
        print(f"   {description}...")

def load_movies():
    movies = []
    if os.path.exists('data/movies.json'):
        with open('data/movies.json', 'r') as f:
            movies = json.load(f)["movies"]
    return movies
