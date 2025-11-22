#!/usr/bin/env python3

import argparse
import lib.semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")
    verify_parser = subparsers.add_parser("verify", help="verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_text_parser.add_argument("text", help="text")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify embeddings")
    embedquery_parser = subparsers.add_parser("embedquery", help="embed query")
    embedquery_parser.add_argument("query", help="query text")
    search_parser = subparsers.add_parser("search", help="search")
    search_parser.add_argument("query", help="query text")
    search_parser.add_argument("--limit", type=int, default=5, help="results limit")
    chunk_parser = subparsers.add_parser("chunk", help="chunk")
    chunk_parser.add_argument("text", help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="chunk size")

    args = parser.parse_args()
    match args.command:
        case "verify":
            lib.semantic_search.verify_model()

        case "embed_text":
            lib.semantic_search.embed_text(args.text)

        case "verify_embeddings":
            lib.semantic_search.verify_embeddings()

        case "embedquery":
            lib.semantic_search.embed_query_text(args.query)

        case "search":
            lib.semantic_search.search_command(args.query, args.limit)

        case "chunk":
            lib.semantic_search.chunk_command(args.text, args.chunk_size)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
