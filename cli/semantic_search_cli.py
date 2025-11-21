#!/usr/bin/env python3

import argparse
import lib.semantic_search

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="available commands")
    verify_parser = subparsers.add_parser("verify", help="verify model")
    embed_text_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_text_parser.add_argument("text", help="text")

    args = parser.parse_args()
    match args.command:
        case "verify":
            lib.semantic_search.verify_model()

        case "embed_text":
            lib.semantic_search.embed_text(args.text)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
