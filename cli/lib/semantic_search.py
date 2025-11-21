#!/usr/bin/env python3

from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")

        embeddings = self.model.encode([text])
        return embeddings[0]

# end class SemanticSearch

def verify_model():
    search = SemanticSearch()

    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
