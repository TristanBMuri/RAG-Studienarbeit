import pytest
from langchain.docstore.document import Document

from complete_pipeline.Rag_fuse import RagFuse


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
        # Create simple embeddings based on text lengths
        embeddings = [[float(len(d.page_content))] for d in docs]
        return embeddings, docs

    def initialize_store(self, embeddings, docs):
        self.embedded = True
        self.embeddings = embeddings
        self.docs = docs
        self.store = True

    def query(self, query, k=3):
        # Return one dummy document
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
    def span(self, name, input=None):
        return self

    def end(self, output=None):
        pass

    def generation(self, name, model, prompt, input):
        return self

    def update(self, **kwargs):
        pass


@pytest.fixture(autouse=True)
def disable_langfuse(monkeypatch):
    # Monkeypatch VectorStoreManager to use DummyVSM
    monkeypatch.setattr(RagFuse, 'VectorStoreConfig', lambda collection_name, persist_directory: None)
    monkeypatch.setattr(RagFuse, 'VectorStoreManager', lambda cfg: DummyVSM())
    # Monkeypatch Langfuse.trace and get_prompt
    class LF:
        def trace(self, name, input, tags=None):
            return DummyTrace()

        def get_prompt(self, template):
            return ""

        def flush(self):
            pass

    monkeypatch.setattr(RagFuse, 'Langfuse', lambda: LF())


def test_init_without_db():
    rag = RagFuse()
    assert rag.vsm is None


def test_load_vector_db(tmp_path):
    # Provide a dummy directory
    dir_path = tmp_path / 'dbdir'
    rag = RagFuse(vector_store_dir=str(dir_path), language="en")
    assert isinstance(rag.vsm, DummyVSM)
    assert rag.vsm.loaded is True


def test_preprocess_skip():
    rag = RagFuse()
    original = rag.md_folder
    rag.preprocess(settings="skip")
    assert rag.md_folder == original


def test_embed_reads_md_and_inits(tmp_path, caplog):
    # Create markdown files
    md_dir = tmp_path / 'md'
    md_dir.mkdir()
    f1 = md_dir / 'a_de.md'
    f1.write_text("hello")
    f2 = md_dir / 'b_en.md'
    f2.write_text("world!!!")

    db_dir = tmp_path / 'db'
    rag = RagFuse(vector_store_dir=str(db_dir), language="de")
    # Override md_folder and vsm
    vsm = DummyVSM()
    rag.vsm = vsm
    rag.md_folder = md_dir
    rag.embed(source_dir=str(md_dir))

    assert vsm.embedded is True
    # Expect two docs
    assert len(vsm.docs) == 2
    # Metadata languages inferred
    names = [d.metadata['file_name'] for d in vsm.docs]
    assert 'a_de.md' in names and 'b_en.md' in names


def test_prompt_no_store(capsys):
    rag = RagFuse()
    result = rag.prompt(prompt="hi")
    captured = capsys.readouterr()
    assert "Vector store not initialized" in captured.out
    assert result is None


def test_prompt_success(monkeypatch):
    rag = RagFuse()
    # Setup VSM with a store
    vsm = DummyVSM()
    vsm.store = True
    rag.vsm = vsm
    # Monkeypatch chat and langfuse
    rag.chat = DummyChat()
    rag.langfuse = type('LF', (), {'trace': lambda self, name, input, tags=None: DummyTrace(), 'get_prompt': lambda self, t: '' ,'flush': lambda self: None})()

    response = rag.prompt(prompt="query")
    assert isinstance(response, DummyResponse)
    assert response.content == "dummy-answer"
