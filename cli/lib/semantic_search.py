#!/usr/bin/env python3

import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
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

        np.save(file='cache/movie_embeddings.npy', arr=self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        movies = []
        for document in documents:
            self.document_map[document['id']] = document
            movies.append(f"{document['title']}: {document['description']}")

        if os.path.exists('cache/movie_embeddings.npy'):
            with open('cache/movie_embeddings.npy', 'rb') as f:
                self.embeddings = np.load(f)

        if self.embeddings is not None and len(self.embeddings) == len(documents):
            return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
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
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query, limit):
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
