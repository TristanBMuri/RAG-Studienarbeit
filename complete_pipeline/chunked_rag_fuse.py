from pathlib import Path
import os

# Import base class and helper functions
from complete_pipeline.Rag_fuse import RagFuse
from vector_db.chroma import (
    embed_chunks_from_directory,
    initialize_chunked_vector_db_langchain,
    load_vector_db_langchain,
    query_documents_by_max_chunk
)

class ChunkedRagFuse(RagFuse):
    """
    RAG subclass that works with a chunked vector database.
    Inherits from RagFuse but overrides methods to use chunked retrieval and storage.
    """
    def __init__(
        self,
        custom_handler=None,
        prompt_template=None,
        vector_store_dir=None,
        language="de"
    ):
        # Initialize base RagFuse without an original vectorstore
        super().__init__(
            custom_handler=custom_handler,
            prompt_template=prompt_template,
            vector_store_dir=None,
            language=language
        )
        self.language = language
        # Determine chunked DB directory
        self.chunked_db_dir = vector_store_dir or f"./chroma_db_{language}_chunks"
        # Try loading existing chunked vectorstore
        if os.path.exists(self.chunked_db_dir):
            print(f"Loading existing chunked vector DB from {self.chunked_db_dir}")
            self.load_chunked_vector_db(language, self.chunked_db_dir)
        else:
            print(f"No existing chunked DB found at {self.chunked_db_dir}, will create on embed.")
        self.chunked_vectorstore = getattr(self, 'chunked_vectorstore', None)

    def load_chunked_vector_db(self, language, persist_directory):
        """
        Load an existing chunked vector database.
        """
        collection_name = f"embeddings_collection_{language}_chunks"
        self.chunked_vectorstore = load_vector_db_langchain(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print(f"Chunked vector DB loaded for {language} at {persist_directory}")

    def embed(
        self,
        language,
        source_dir="src/company_pdfs"
    ):
        """
        Embed documents into a chunked vector database.
        Uses initialize_chunked_vector_db_langchain from chroma.py.
        """
        if source_dir:
            self.md_folder = Path(source_dir)
        embeddings_array, md_documents = embed_chunks_from_directory(self.md_folder)
        self.chunked_vectorstore = initialize_chunked_vector_db_langchain(
            embeddings_array,
            md_documents,
            collection_name=f"embeddings_collection_{language}_chunks",
            persist_directory=self.chunked_db_dir
        )
        print(f"Created new chunked vector DB at {self.chunked_db_dir}")
        return self.chunked_vectorstore

    def prompt(
        self,
        prompt=None,
        custom_handler=None,
        k=50,
        top_n=3
    ):
        """
        Override prompt to perform retrieval over chunks.
        Queries the chunked vector database and groups by document to pick top-N.
        """
        query_input = prompt or input("Enter your question: ")
        # Ensure base uses our chunked store
        self.vectorstore = self.chunked_vectorstore
        raw_results = query_documents_by_max_chunk(
            self.chunked_vectorstore,
            query_input,
            k=k,
            top_n=top_n
        )
        documents = [doc for doc, score in raw_results]
        context = "\n".join([doc.page_content for doc in documents])
        # Optionally, inject context into prompt logic here
        return super().prompt(prompt=query_input, custom_handler=custom_handler)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Chunked RAG Fuse: load or embed documents into a chunked vector DB"
    )
    parser.add_argument(
        "--input_dir", type=str, default="./src/company_pdfs",
        help="Path to markdown (.md) documents folder to chunk and embed"
    )
    parser.add_argument(
        "--language", type=str, default="de",
        help="Language code for filtering documents"
    )
    parser.add_argument(
        "--db_dir", type=str, default=None,
        help="Directory to persist or load the chunked DB"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    language = args.language
    db_dir = args.db_dir or f"./chroma_db_{language}_chunks"

    rag = ChunkedRagFuse(
        prompt_template="combined_rag",
        vector_store_dir=db_dir,
        language=language
    )

    if rag.chunked_vectorstore is None:
        print(f"Embedding markdown chunks from {input_dir} into new chunked DB...")
        rag.embed(language=language, source_dir=input_dir)
    else:
        print(f"Using loaded chunked DB from {db_dir}")

    # prompt
    question = "Welche Versicherungen beteiligen sich an der Finanzierung von Hilfsmitteln und Krankentransporten?"
    answer = rag.prompt(prompt=question)
    print("Answer:\n", answer)
