#!/usr/bin/env python3

import argparse
import json
import math
import string

from inverted_index import InvertedIndex
from nltk.stem import PorterStemmer

def main() -> None:
    remove_punctuation = str.maketrans('', '', string.punctuation)

    with open('data/movies.json', 'r') as f:
        movies_data = json.load(f)
    movies = sorted(movies_data['movies'], key=lambda x: x['id'])

    with open('data/stopwords.txt', 'r') as f:
        stop_words = f.read().splitlines()

    stemmer = PorterStemmer()
    inverted_index = InvertedIndex()

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build search index")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("document", help="document id")
    tf_parser.add_argument("term", help="term to count in a document")
    idf_parser = subparsers.add_parser("idf", help="inverse document frequency")
    idf_parser.add_argument("term", help="term to count in all documents")

    args = parser.parse_args()
    match args.command:
        case "search":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Error loading inverted index: {e}")
                return

            query = args.query.lower().translate(remove_punctuation)
            query_tokens = [token for token in query.split() if token]
            query_tokens = [token for token in query_tokens if token not in stop_words]
            query_tokens = [stemmer.stem(token) for token in query_tokens]

            print(f"Searching for: {args.query}")
            results = []
            for token in query_tokens:
                token_docs = inverted_index.get_documents(token)
                results.extend(token_docs)
                if len(results) >= 5:
                    break

            for doc_id in results[:5]:
                document = inverted_index.get_document(doc_id)
                print(f"{document['id']:4}: {document['title']}")

        case "tf":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Error loading inverted index: {e}")
                return

            try:
                print(inverted_index.get_tf(args.document, args.term))
            except Exception as e:
                print(e)
                return

        case "idf":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Error loading inverted index: {e}")
                return

            doc_count, term_doc_count = inverted_index.get_idf(args.term)
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "build":
            inverted_index.build(movies)
            inverted_index.save()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
