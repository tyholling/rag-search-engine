#!/usr/bin/env python3

import argparse
import json
import string

def main() -> None:
    with open('data/movies.json', 'r') as f:
        data = json.load(f)
    movies = sorted(data['movies'], key=lambda x: x['id'])

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    remove_punctuation = str.maketrans('', '', string.punctuation)
    query = args.query.lower().translate(remove_punctuation)
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = list(movie['title']
                           for movie in movies
                           if query in movie['title'].lower().translate(remove_punctuation))
            for i, title in enumerate(results[:5]):
                print(f"{i + 1}. {title}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
