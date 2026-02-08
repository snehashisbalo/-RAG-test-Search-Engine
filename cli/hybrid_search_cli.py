
#!/usr/bin/env python3
import argparse
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -----------------------
    # Normalize command
    # -----------------------
    norm_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    norm_parser.add_argument('scores', type=float, nargs='+', help="List of scores to normalize")

    # -----------------------
    # Weighted search command
    # -----------------------
    ws_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    ws_parser.add_argument('query', type=str, help="Query to search for")
    ws_parser.add_argument('--limit', type=int, default=5, help="Number of results to return")
    ws_parser.add_argument('--alpha', type=float, default=0.5, help="Weight for BM25 vs semantic similarity")


    rrf_parser = subparsers.add_parser("rrf-search", help="Search using Reciprocal Rank Fusion")
    rrf_parser.add_argument("query", type=str, help="Query string")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    
    args = parser.parse_args()

    match args.command:
        case "rrf-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(args.query, k=args.k, limit=args.limit)
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['title']}")
                print(f"   RRF Score: {r['rrf_score']:.3f}")
                print(f"   BM25 Rank: {r['bm25_rank']}, Semantic Rank: {r['sem_rank']}")
                print(f"   {r['description'][:200]}...\n")
        case "normalize":
            norm_scores = normalize_scores(args.scores)
            for score in norm_scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            # TODO: replace this with actual document loading
            documents = []  # list of dicts: {'id','title','description'}
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, alpha=args.alpha, limit=args.limit)

            for idx, r in enumerate(results, start=1):
                print(f"{idx}. {r['title']}")
                print(f"   Hybrid Score: {r['hybrid_score']:.3f}")
                print(f"   BM25: {r['bm25_score']:.3f}, Semantic: {r['sem_score']:.3f}")
                print(f"   {r['description'][:200]}...\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()



