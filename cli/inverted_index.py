import os
import pickle
import string

from collections import Counter
from nltk.stem import PorterStemmer

class InvertedIndex:

    def __init__(self):
        # dictionary mapping tokens to sets of document ids
        self.index = {}
        # dictionary mapping document ids to document objects
        self.docmap = {}
        # dictionary mapping documents ids to counter objects
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        stemmer = PorterStemmer()
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokens = text.lower().translate(remove_punctuation).split()

        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            stem = stemmer.stem(token)

            if stem not in self.index:
                self.index[stem] = set()
            self.index[stem].add(doc_id)

            self.term_frequencies[doc_id][stem] += 1

    def get_documents(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def get_document(self, doc_id):
        return self.docmap[doc_id]

    def get_tf(self, doc_id, term):
        stemmer = PorterStemmer()
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokens = term.lower().translate(remove_punctuation).split()
        if len(tokens) != 1:
            raise Exception("expected one token")
        token = stemmer.stem(tokens[0])

        if int(doc_id) not in self.term_frequencies:
            return 0
        return self.term_frequencies[int(doc_id)][token]

    def get_idf(self, term):
        stemmer = PorterStemmer()
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokens = term.lower().translate(remove_punctuation).split()
        if len(tokens) != 1:
            raise Exception("expected one token")
        token = stemmer.stem(tokens[0])

        # how many documents total
        doc_count = len(self.docmap)
        # how many documents contain a term
        term_doc_count = len(list(self.index[token])) if token in self.index else 0
        return doc_count, term_doc_count

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

    def load(self):
        if not os.path.exists('cache'):
            raise FileNotFoundError("Cache directory does not exist")
        if not os.path.exists('cache/index.pkl'):
            raise FileNotFoundError("Index file does not exist")
        if not os.path.exists('cache/docmap.pkl'):
            raise FileNotFoundError("Docmap file does not exist")
        if not os.path.exists('cache/term_frequencies.pkl'):
            raise FileNotFoundError("Term frequencies file does not exist")

        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open('cache/term_frequencies.pkl', 'rb') as f:
            self.term_frequencies = pickle.load(f)
