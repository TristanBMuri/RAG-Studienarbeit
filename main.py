import argparse
from pathlib import Path

from complete_pipeline.Rag_fuse import RagFuse
from complete_pipeline.chunked_rag_fuse import ChunkedRagFuse


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline: full or chunked retrieval"
    )
    parser.add_argument(
        "--chunked", action="store_true",
        help="Use chunked vector database (default: full-document DB)"
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory of source documents (PDFs or Markdown)"
    )
    parser.add_argument(
        "--db-dir", type=Path, required=True,
        help="Directory to load/persist the vector database"
    )
    parser.add_argument(
        "--language", type=str, default="de",
        help="Language code for embedding filter"
    )
    parser.add_argument(
        "--template", type=str,
        help="Name of the Langfuse prompt template (default: rag_prompt)"
    )
    parser.add_argument(
        "--question", type=str, required=True,
        help="The user query to ask the RAG system"
    )
    parser.add_argument(
        "--force-embed", action="store_true",
        help="Re-index/refresh the vector database before querying"
    )
    args = parser.parse_args()

    if args.chunked:
        rag = ChunkedRagFuse(
            custom_handler=None,
            prompt_template=args.template,
            vector_store_dir=str(args.db_dir),
            language=args.language,
            source_dir=str(args.input_dir)
        )
    else:
        rag = RagFuse(
            custom_handler=None,
            prompt_template=args.template,
            vector_store_dir=str(args.db_dir),
            language=args.language
        )

    # Optionally force re-embedding (overwrites or initializes DB)
    if args.force_embed:
        try:
            print("Embedding documents to vector DB...")
            rag.embed(source_dir=str(args.input_dir))
        except Exception as e:
            print(f"Error during embedding: {e}")
            return

    # Execute the prompt/query
    try:
        answer = rag.prompt(prompt=args.question)
        print("\n=== Answer ===")
        print(answer.content if hasattr(answer, 'content') else answer)
    except Exception as e:
        print(f"Error during prompt: {e}")


if __name__ == "__main__":
    main()
