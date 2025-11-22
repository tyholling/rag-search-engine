#!/usr/bin/env python3

import argparse

import lib.hybrid_search as hybrid_search

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="normalize")
    normalize_parser.add_argument("scores", nargs="+", type=float, help="scores")

    args = parser.parse_args()
    match args.command:
        case "normalize":
            hybrid_search.normalize_command(args.scores)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
