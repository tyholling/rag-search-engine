class InvertedIndex:

    def __init__(self):
        # dictionary mapping tokens to sets of document ids
        self.index = {}
        # dictionary mapping document ids to document objects
        self.docmap = {}

    def __add_document(self, doc_id, text):
        import string
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        remove_punctuation = str.maketrans('', '', string.punctuation)
        tokens = text.lower().translate(remove_punctuation).split()
        tokens = [stemmer.stem(token) for token in tokens]
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

    def get_document(self, doc_id):
        return self.docmap[doc_id]

    def build(self, movies):
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie
    
    def save(self):
        import os
        import pickle

        # Create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)

        # Save index to cache/index.pkl
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)

        # Save docmap to cache/docmap.pkl
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)

    def load(self):
        import os
        import pickle

        # Check if cache directory exists
        if not os.path.exists('cache'):
            raise FileNotFoundError("Cache directory does not exist")

        # Check if index file exists
        if not os.path.exists('cache/index.pkl'):
            raise FileNotFoundError("Index file does not exist")

        # Check if docmap file exists
        if not os.path.exists('cache/docmap.pkl'):
            raise FileNotFoundError("Docmap file does not exist")

        # Load index from cache/index.pkl
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)

        # Load docmap from cache/docmap.pkl
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
