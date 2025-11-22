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
