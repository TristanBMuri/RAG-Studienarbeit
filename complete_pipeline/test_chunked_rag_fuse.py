import sys, pathlib
# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
from pathlib import Path

# Stub out vector_store_manager so import works
import types, sys
sys.modules['vector_store_manager'] = types.SimpleNamespace(
    VectorStoreManager=lambda cfg: None,
    VectorStoreConfig=lambda *args, **kwargs: None
)

# Import the module under test
import complete_pipeline.chunked_rag_fuse as chunked_module
from complete_pipeline.chunked_rag_fuse import ChunkedRagFuse
from langchain.docstore.document import Document

# Dummy manager for chunked embeddings
class DummyChunkManager:
    def __init__(self):
        self.loaded = False
        self.initialized = False
        self.store = False
        self.chunks = None

    def load_store(self):
        self.loaded = True
        self.store = True

    def embed_and_chunk(self, texts):
        docs = [Document(page_content=t, metadata={"source": i}) for i, t in enumerate(texts)]
        embeddings = [[float(len(d.page_content))] for d in docs]
        return embeddings, docs

    def initialize_store(self, embeddings, docs):
        self.initialized = True
        self.chunks = docs
        self.store = True

    def query(self, query, k=3):
        return [(Document(page_content="doc", metadata={}), 0.2)]

# Monkeypatch fixtures
@pytest.fixture(autouse=True)
def patch_manager_and_preprocess(monkeypatch, tmp_path):
    # Patch VectorStoreConfig and VectorStoreManager in chunked_module
    monkeypatch.setattr(chunked_module, 'VectorStoreConfig', lambda *args, **kwargs: None)
    monkeypatch.setattr(chunked_module, 'VectorStoreManager', lambda cfg: DummyChunkManager())
    # Prevent actual preprocessing (no side effects)
    monkeypatch.setattr(chunked_module.ChunkedRagFuse, 'preprocess', lambda self, settings, input_dir, output_dir, preprocessor: None)

# Tests

def test_init_with_existing_db(tmp_path):
    db_dir = tmp_path / 'existing_db'
    db_dir.mkdir()
    loader = ChunkedRagFuse(vector_store_dir=str(db_dir), language='en', source_dir=str(tmp_path / 'src'))
    # Should have loaded on init
    assert isinstance(loader.vsm, DummyChunkManager)
    assert loader.chunk_vsm.loaded is True
    assert loader.chunk_vsm.store is True


def test_embed_creates_and_initializes_store(tmp_path):
    db_dir = tmp_path / 'new_db'
    src_dir = tmp_path / 'src_md'
    src_dir.mkdir()
    md_out = db_dir / 'md'
    md_out.mkdir(parents=True)
    (md_out / 'one.md').write_text('AAA')
    (md_out / 'two.md').write_text('BBBB')

    ref = ChunkedRagFuse(vector_store_dir=str(db_dir), language='en', source_dir=str(src_dir))
    assert not ref.chunk_vsm.initialized
    ref.embed(source_dir=str(src_dir))
    assert ref.chunk_vsm.initialized is True
    assert ref.vsm is ref.chunk_vsm
    assert len(ref.chunk_vsm.chunks) == 2


def test_prompt_triggers_embed_and_calls_super(monkeypatch, tmp_path):
    db_dir = tmp_path / 'prompt_db'
    src_dir = tmp_path / 'src_md2'
    src_dir.mkdir()
    # Create existing db directory so load_store is invoked in __init__
    md_out = db_dir / 'md'
    md_out.mkdir(parents=True)
    (md_out / 'm.md').write_text('XYZ')

    calls = {}
    def fake_super_prompt(self, prompt=None, custom_handler=None):
        calls['ran'] = True
        return 'fake-answer'
    monkeypatch.setattr(chunked_module.RagFuse, 'prompt', fake_super_prompt)

    ref = ChunkedRagFuse(vector_store_dir=str(db_dir), language='en', source_dir=str(src_dir))
    # Since the DB existed at init, chunk_vsm.store should be True
    assert ref.chunk_vsm.store is True

    answer = ref.prompt(prompt='hi')
    # Should delegate to super.prompt
    assert calls.get('ran', False) is True
    assert answer == 'fake-answer'
