#!/usr/bin/env python3

import argparse
import json

def main() -> None:
    with open('data/movies.json', 'r') as f:
        data = json.load(f)
    movies = sorted(data['movies'], key=lambda x: x['id'])

    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = list(movie['title'] for movie in movies if args.query in movie['title'])
            for i, title in enumerate(results[:5]):
                print(f"{i + 1}. {title}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
