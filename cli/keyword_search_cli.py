#!/usr/bin/env python3

import argparse
import json
import math
import string

from inverted_index import InvertedIndex, BM25_K1
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
    subparsers = parser.add_subparsers(dest="command", help="available commands")
    search_parser = subparsers.add_parser("search", help="search movies")
    search_parser.add_argument("query", type=str, help="search query")
    subparsers.add_parser("build", help="build search index")
    tf_parser = subparsers.add_parser("tf", help="term frequency")
    tf_parser.add_argument("doc_id", help="document id")
    tf_parser.add_argument("term", help="term to count in a document")
    idf_parser = subparsers.add_parser("idf", help="inverse document frequency")
    idf_parser.add_argument("term", help="term to count in all documents")
    tfidf_parser = subparsers.add_parser("tfidf", help="calculate tf-idf")
    tfidf_parser.add_argument("doc_id", help="document id")
    tfidf_parser.add_argument("term", help="term to analyze for a document")
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
                print(inverted_index.get_tf(args.doc_id, args.term))
            except Exception as e:
                print(e)

        case "idf":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Error loading inverted index: {e}")
                return

            doc_count, term_doc_count = inverted_index.get_idf(args.term)
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")

        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")

        case "tfidf":
            try:
                inverted_index.load()
            except Exception as e:
                print(f"Error loading inverted index: {e}")
                return

            try:
                stemmer = PorterStemmer()
                tokens = args.term.lower().split()
                if len(tokens) != 1:
                    raise Exception("expected one token")
                token = stemmer.stem(tokens[0])

                tf = inverted_index.get_tf(args.doc_id, token)
                doc_count, term_doc_count = inverted_index.get_idf(token)
                idf = math.log((doc_count + 1) / (term_doc_count + 1))
                tf_idf = tf * idf
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except Exception as e:
                print(e)

        case "build":
            inverted_index.build(movies)
            inverted_index.save()

        case _:
            parser.print_help()

def bm25_idf_command(term: str) -> float:
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
    except Exception as e:
        print(f"Error loading inverted index: {e}")
        return 0

    return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1):
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
    except Exception as e:
        print(f"Error loading inverted index: {e}")
        return 0

    return inverted_index.get_bm25_tf(doc_id, term, k1)

if __name__ == "__main__":
    main()
