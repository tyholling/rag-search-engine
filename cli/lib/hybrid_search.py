import json
import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:

    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        with open('data/movies.json', 'r') as f:
            movies_data = json.load(f)
        movies = sorted(movies_data['movies'], key=lambda x: x['id'])
        self.idx.build(movies)
        self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

# end class HybridSearch
