import pytest
from pathlib import Path

# Import the module under test
import complete_pipeline.Rag_fuse as rag_module
from complete_pipeline.Rag_fuse import RagFuse
from langchain.docstore.document import Document

# Dummy implementations to isolate RagFuse logic
class DummyVSM:
    def __init__(self):
        self.loaded = False
        self.embedded = False
        self.embeddings = None
        self.docs = None
        self.store = False

    def load_store(self):
        self.loaded = True
        self.store = True

    def embed_documents(self, docs):
        embeddings = [[float(len(d.page_content))] for d in docs]
        return embeddings, docs

    def initialize_store(self, embeddings, docs):
        self.embedded = True
        self.embeddings = embeddings
        self.docs = docs
        self.store = True

    def query(self, query, k=3):
        return [(Document(page_content="doc", metadata={}), 0.1)]

class DummyResponse:
    def __init__(self):
        self.content = "dummy-answer"
        self.usage = {"input_tokens": 1, "output_tokens": 1}

class DummyChat:
    def get_model(self):
        return "model"
    def prompt(self, query, qr, session_id, full_response):
        return DummyResponse()

class DummyTrace:
    def span(self, name, input=None): return self
    def end(self, output=None, usage_details=None): pass
    def generation(self, name, model, prompt, input): return self
    def update(self, **kwargs): pass

@pytest.fixture(autouse=True)
def disable_external_dependencies(monkeypatch):
    # Patch VectorStoreConfig and VectorStoreManager
    monkeypatch.setattr(rag_module, 'VectorStoreConfig', lambda collection_name, persist_directory: None)
    monkeypatch.setattr(rag_module, 'VectorStoreManager', lambda cfg: DummyVSM())
    # Patch Langfuse to no-op
    class LF:
        def trace(self, name, input, tags=None): return DummyTrace()
        def get_prompt(self, template): return ""
        def flush(self): pass
    monkeypatch.setattr(rag_module, 'Langfuse', lambda: LF())


def test_init_without_db():
    rag = RagFuse()
    assert rag.vsm is None


def test_load_vector_db(tmp_path):
    dir_path = tmp_path / 'dbdir'
    rag = RagFuse(vector_store_dir=str(dir_path), language="en")
    assert isinstance(rag.vsm, DummyVSM)
    assert rag.vsm.loaded is True


def test_preprocess_skip():
    rag = RagFuse()
    original = rag.md_folder
    rag.preprocess(settings="skip")
    assert rag.md_folder == original


def test_embed_reads_md_and_inits(tmp_path):
    md_dir = tmp_path / 'md'
    md_dir.mkdir()
    (md_dir / 'a_de.md').write_text("hello")
    (md_dir / 'b_en.md').write_text("world!!!")
    rag = RagFuse(vector_store_dir=str(tmp_path / 'db'), language="de")
    vsm = DummyVSM()
    rag.vsm = vsm
    rag.md_folder = md_dir
    rag.embed(source_dir=str(md_dir))
    assert vsm.embedded is True
    assert len(vsm.docs) == 2
    names = [d.metadata['file_name'] for d in vsm.docs]
    assert 'a_de.md' in names and 'b_en.md' in names


def test_prompt_no_store(capsys):
    rag = RagFuse()
    result = rag.prompt(prompt="hi")
    captured = capsys.readouterr()
    assert "Vector store not initialized" in captured.out
    assert result is None


def test_prompt_success():
    rag = RagFuse()
    vsm = DummyVSM()
    vsm.store = True
    rag.vsm = vsm
    rag.chat = DummyChat()
    rag.langfuse = type('LF', (), {
        'trace': lambda self, name, input, tags=None: DummyTrace(),
        'get_prompt': lambda self, t: '',
        'flush': lambda self: None
    })()
    response = rag.prompt(prompt="query", full_doc=True)
    assert hasattr(response, 'content') and response.content == "dummy-answer"

