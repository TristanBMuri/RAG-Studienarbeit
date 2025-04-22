from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorStoreConfig:
    """
    Configuration for vector store initialization.

    Attributes:
        collection_name: Base name of the Chroma collection.
        persist_directory: Path to persist the database.
        embedding_model: Name of the HuggingFace embedding model.
        batch_size: Number of vectors to add per batch.
        chunk_size: Character length for chunking (if chunked mode).
        chunk_overlap: Overlap size between chunks.
    """
    def __init__(
        self,
        collection_name: str,
        persist_directory: Union[str, Path],
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 160,
        chunk_size: int = 250,
        chunk_overlap: int = 30,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap


class VectorStoreManager:
    """
    Manages Chroma vector stores: initialization, loading, embedding, and querying.
    Supports full-document and chunked-document modes.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.embedding_function = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.store: Optional[Chroma] = None

    def initialize_store(
        self,
        embeddings: np.ndarray,
        documents: List[Document],
        chunked: bool = False
    ) -> Chroma:
        """
        Create a new Chroma store and add embeddings/documents in batches.
        If chunked=True, expects `documents` to be already split into chunks.
        """
        store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.config.persist_directory
        )
        total = len(documents)
        for start in range(0, total, self.config.batch_size):
            end = min(start + self.config.batch_size, total)
            batch_docs = documents[start:end]
            ids = [f"vec_{i+start}" for i in range(len(batch_docs))]
            embeddings_batch = embeddings[start:end].tolist()
            metadatas = [doc.metadata for doc in batch_docs]
            texts = [doc.page_content for doc in batch_docs]
            store._collection.add(
                ids=ids,
                embeddings=embeddings_batch,
                metadatas=metadatas,
                documents=texts
            )
        self.store = store
        return store

    def load_store(self) -> Chroma:
        """
        Load an existing Chroma store from the configured directory.
        """
        store = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.config.persist_directory
        )
        self.store = store
        return store

    def embed_documents(
        self,
        documents: List[Document]
    ) -> (np.ndarray, List[Document]):
        """
        Compute embeddings for full documents.
        Returns embeddings array and the same documents list.
        """
        contents = [doc.page_content for doc in documents]
        embeddings = self.embedding_function.embed_documents(contents)
        return np.array(embeddings), documents

    def embed_and_chunk(
        self,
        raw_texts: List[str]
    ) -> (np.ndarray, List[Document]):
        """
        Split raw_texts into chunks, compute embeddings, and return chunked documents.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        chunks: List[Document] = []
        for idx, text in enumerate(raw_texts):
            text_chunks = splitter.split_text(text)
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={"source_index": idx, "chunk_index": i}
                ))
        embeddings = self.embedding_function.embed_documents([
            c.page_content for c in chunks
        ])
        return np.array(embeddings), chunks

    def query(self, query: str, k: int = 3):
        """
        Perform a similarity search on the loaded store.
        """
        if not self.store:
            raise ValueError("Store not initialized. Call load_store or initialize_store first.")
        return self.store.similarity_search_with_score(query, k=k)
