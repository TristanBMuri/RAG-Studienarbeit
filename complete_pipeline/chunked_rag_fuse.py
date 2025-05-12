from pathlib import Path

from numpy.f2py.auxfuncs import throw_error

from complete_pipeline.Rag_fuse import RagFuse
from vector_db.vector_store_manager import VectorStoreConfig, VectorStoreManager


class ChunkedRagFuse(RagFuse):
    def __init__(
        self,
        custom_handler=None,
        prompt_template=None,
        vector_store_dir=None,
        language="de",
        source_dir="./src/company_pdfs"
    ):
        super().__init__(custom_handler=custom_handler,
                         prompt_template=prompt_template,
                         vector_store_dir=None,
                         language=language)
        # paths
        module_root = Path(__file__).resolve().parent.parent.parent
        self.source_dir = module_root / source_dir
        self.chunked_db_dir = Path(vector_store_dir) if vector_store_dir else module_root / f"chroma_db_{language}_chunks"
        # vector store manager for chunks
        cfg = VectorStoreConfig(
            collection_name=f"embeddings_collection_{language}_chunks",
            persist_directory=str(self.chunked_db_dir)
        )
        self.chunk_vsm = VectorStoreManager(cfg)
        if self.chunked_db_dir.exists():
            print(f"Loading existing chunked DB from {self.chunked_db_dir}")
            self.chunk_vsm.load_store()
            self.vsm = self.chunk_vsm
        else:
            print(f"No chunked DB at {self.chunked_db_dir}; will embed on first prompt")

    def embed(self, source_dir: str = None):
        src = Path(source_dir) if source_dir else self.source_dir
        # preprocess to markdown
        temp_md = self.chunked_db_dir / "md"
        temp_md.mkdir(parents=True, exist_ok=True)
        self.preprocess(settings="d", input_dir=src, output_dir=temp_md, preprocessor="md")
        # read and chunk
        docs = self._md_glob(temp_md)
        embeddings, chunks = self.chunk_vsm.embed_and_chunk(docs)
        self.chunk_vsm.initialize_store(embeddings, chunks)
        self.vsm = self.chunk_vsm
        return self.vsm

    def prompt(
        self,
        prompt=None,
        custom_handler=None,
        k=50,
        top_n=3,
        full_doc: bool = True
    ):
        query_input = prompt
        if not query_input:
            throw_error("Please provide a query.")
        if not self.chunk_vsm.store:
            print(f"Embedding new chunked DB from '{self.source_dir}'...")
            self.embed(source_dir=str(self.source_dir))
        self.vsm = self.chunk_vsm
        return super().prompt(prompt=query_input, custom_handler=custom_handler, top_n=top_n, k=k, full_doc=full_doc)
