class InvertedIndex:

    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = text.lower().split()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(list(self.index[term]))
        return []

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
