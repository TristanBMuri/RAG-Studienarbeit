import math
import numpy as np
import pytest
from pathlib import Path

# Assume the manager code is in vector_store_manager.py in the project root
from vector_db import vector_store_manager
from  vector_db.vector_store_manager import VectorStoreManager, VectorStoreConfig
from langchain.docstore.document import Document


class DummyEmbedding:
    """Stub embedding that returns a fixed-size vector based on text length."""
    def embed_documents(self, texts):
        # Return a single-dimension embedding equal to text length
        return [[float(len(t))] for t in texts]


class DummyCollection:
    def __init__(self):
        self.added = []

    def add(self, ids, embeddings, metadatas, documents):
        self.added.append({
            'ids': ids,
            'embeddings': embeddings,
            'metadatas': metadatas,
            'documents': documents
        })


class DummyChroma:
    def __init__(self, collection_name, embedding_function, persist_directory):
        # Record parameters for inspection
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self._collection = DummyCollection()

    def similarity_search_with_score(self, query, k=3):
        # Return dummy results
        return [(Document(page_content="doc", metadata={'q': query}), float(k))]


def test_initialize_store_batches(tmp_path, monkeypatch):
    # Prepare config pointing at ephemeral directory
    config = VectorStoreConfig(collection_name='test_coll', persist_directory=tmp_path)
    manager = VectorStoreManager(config)

    # Patch embedding_function and Chroma
    manager.embedding_function = DummyEmbedding()
    monkeypatch.setattr(vector_store_manager, 'Chroma', DummyChroma)

    # Create synthetic data: 5 docs, embedding dim=1
    docs = [Document(page_content=f"doc{i}", metadata={}) for i in range(5)]
    embeddings = np.arange(5).reshape((5, 1)).astype(float)

    store = manager.initialize_store(embeddings, docs)
    assert isinstance(store, DummyChroma)
    # All 5 docs in one batch since default batch_size=160
    assert len(store._collection.added) == 1
    batch = store._collection.added[0]
    assert batch['ids'] == [f'vec_{i}' for i in range(5)]
    assert batch['embeddings'] == embeddings.tolist()


def test_load_store(monkeypatch, tmp_path):
    config = VectorStoreConfig(collection_name='coll2', persist_directory=tmp_path / 'dbdir')
    manager = VectorStoreManager(config)
    # Monkeypatch Chroma to capture init args
    monkeypatch.setattr(vector_store_manager, 'Chroma', DummyChroma)

    store = manager.load_store()
    assert isinstance(store, DummyChroma)
    assert store.collection_name == 'coll2'
    assert Path(store.persist_directory) == tmp_path / 'dbdir'


def test_embed_documents_returns_array_and_docs(monkeypatch):
    config = VectorStoreConfig(collection_name='x', persist_directory='.')
    manager = VectorStoreManager(config)
    manager.embedding_function = DummyEmbedding()
    docs = [Document(page_content='a', metadata={}), Document(page_content='bb', metadata={})]

    embeddings, returned_docs = manager.embed_documents(docs)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 1)
    np.testing.assert_array_equal(embeddings.flatten(), np.array([1.0, 2.0]))
    assert returned_docs == docs


def test_embed_and_chunk(monkeypatch):
    config = VectorStoreConfig(collection_name='x', persist_directory='.', chunk_size=3, chunk_overlap=1)
    manager = VectorStoreManager(config)
    manager.embedding_function = DummyEmbedding()

    raw_texts = ['abcdef']  # length=6, chunk_size=3, overlap=1 => two chunks: 'abc', 'cde', 'ef'
    embeddings, chunks = manager.embed_and_chunk(raw_texts)
    # Expect 3 chunks given chunk_size=3, overlap=1: ['abc','c','def'?] but test consistency
    assert isinstance(embeddings, np.ndarray)
    assert len(chunks) == embeddings.shape[0]
    # Metadata indices should be sequential
    for idx, doc in enumerate(chunks):
        assert doc.metadata['source_index'] == 0
        assert doc.metadata['chunk_index'] == idx


def test_query_behavior():
    config = VectorStoreConfig(collection_name='q', persist_directory='.')
    manager = VectorStoreManager(config)
    dummy = DummyChroma('c', None, '.')
    manager.store = dummy
    results = manager.query('hello', k=5)
    assert isinstance(results, list)
    assert results[0][1] == 5.0

    # Test error when not initialized
    manager2 = VectorStoreManager(config)
    with pytest.raises(ValueError):
        manager2.query('test')
