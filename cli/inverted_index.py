import os
import math
import pickle
import re
import string

from collections import Counter
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:

    def __init__(self):
        # dictionary mapping tokens to sets of document ids
        self.index = {}
        # dictionary mapping document ids to document objects
        self.docmap = {}
        # dictionary mapping document ids to counter objects
        self.term_frequencies = {}
        # dictionary mapping document ids to lengths
        self.doc_lengths = {}

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        # total = sum(self.doc_lengths.values())
        # n = len(self.doc_lengths)
        # return total / n if n > 0 else 0
        return total_length / len(self.doc_lengths)

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        token = tokenize_one(term)
        return sorted(list(self.index[term])) if term in self.index else []

    def get_document(self, doc_id):
        return self.docmap[doc_id]

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_one(term)

        if doc_id not in self.term_frequencies:
            raise Exception
        return self.term_frequencies[int(doc_id)][token]

    def get_idf(self, term: str):
        token = tokenize_one(term)

        # how many documents total
        doc_count = len(self.docmap)
        # how many documents contain a term
        term_doc_count = len(self.index[token])
        return doc_count, term_doc_count

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_one(term)

        n = len(self.docmap)
        if token not in self.index:
            raise Exception("token not in index")
        df = len(self.index[token])
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        token = tokenize_one(term)
        tf = self.get_tf(doc_id, token)

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        tf_component = self.get_bm25_tf(doc_id, term)
        idf_component = self.get_bm25_idf(term)
        return tf_component * idf_component

    def get_bm25(self, doc_id: int, term: str) -> float:
        token = tokenize_one(term)

        bm25_tf = self.get_bm25_tf(doc_id, token)
        bm25_idf = self.get_bm25_idf(token)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = tokenize(query)

        scores = []
        for doc_id, document in self.docmap.items():
            doc_score = 0.0
            for token in tokens:
                doc_score += self.get_bm25(doc_id, token)
            scores.append((doc_id, document['title'], doc_score))
        return sorted(scores, key=lambda x: x[2], reverse=True)[:limit]

    def build(self, movies):
        for movie in movies:
            doc_id = int(movie['id'])
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open('cache/term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open('cache/doc_lengths.pkl', 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not os.path.exists('cache'):
            raise FileNotFoundError("Cache directory does not exist")
        if not os.path.exists('cache/index.pkl'):
            raise FileNotFoundError("Index file does not exist")
        if not os.path.exists('cache/docmap.pkl'):
            raise FileNotFoundError("Docmap file does not exist")
        if not os.path.exists('cache/term_frequencies.pkl'):
            raise FileNotFoundError("Term frequencies file does not exist")
        if not os.path.exists('cache/doc_lengths.pkl'):
            raise FileNotFoundError("Doc lengths does not exist")

        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open('cache/term_frequencies.pkl', 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open('cache/doc_lengths.pkl', 'rb') as f:
            self.doc_lengths = pickle.load(f)

# end class InvertedIndex

def tokenize(text: str) -> list[str]:
    stemmer = PorterStemmer()

    with open('data/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()

    tokens = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    tokens = [token for token in tokens if token]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def tokenize_one(text: str) -> str:
    tokens = tokenize(text)
    if len(tokens) != 1:
        raise Exception("expected one token")
    return tokens[0]
