import os
import time
import shutil
from pathlib import Path
from datetime import datetime
import uuid
import sys

from langfuse import Langfuse
from complete_pipeline.Rag_fuse import RagFuse
from complete_pipeline.chunked_rag_fuse import ChunkedRagFuse
from vector_db.vector_store_manager import VectorStoreConfig, VectorStoreManager

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = BASE_DIR / "Docs"
DB_BASE_DIR = BASE_DIR / "bench_dbs"
DB_BASE_DIR.mkdir(exist_ok=True)
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base",
    "all-MiniLM-L6-v2",
    # add more models as needed
]
VARIANTS = [
    ("chunked", ChunkedRagFuse),
    ("full", RagFuse)
]
LANG = "de"
DATASET_NAME = "rag_conci_test_fragen"

# --- Dataset setup ---
lf = Langfuse()
dataset = lf.get_dataset(DATASET_NAME)

# --- No-op handler to bypass Langfuse tracing errors ---
class DummyHandler:
    def trace(self, *args, **kwargs):
        return DummySpan()

class DummySpan:
    def span(self, *args, **kwargs): return self
    def generation(self, *args, **kwargs): return self
    def end(self, *args, **kwargs): pass

# Stub out flush to avoid hanging
# (we'll not flush real Langfuse to avoid update errors)
# lf.flush = lambda: None

# --- Benchmark loop ---
for model_name in EMBED_MODELS:
    for variant_name, RagClass in VARIANTS:
        # prepare vector DB directory for this run
        db_dir = DB_BASE_DIR / f"{variant_name}_{model_name.replace('/', '_')}"
        if db_dir.exists():
            shutil.rmtree(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        # instantiate RAG (full or chunked)
        kwargs = {
            'custom_handler': None,
            'prompt_template': 'combined_rag',
            'vector_store_dir': str(db_dir),
            'language': LANG
        }
        if variant_name == 'chunked':
            kwargs['source_dir'] = str(PDF_DIR)
        rag = RagClass(**kwargs)

        # set embedding model
        new_cfg = VectorStoreConfig(
            collection_name=f"embeddings_collection_{LANG}",
            persist_directory=str(db_dir),
            embedding_model=model_name,
        )
        rag.vsm = VectorStoreManager(new_cfg)
        if variant_name == 'chunked':
            rag.chunk_vsm = rag.vsm  # keep both refs in sync

        # embed documents and load DB
        if variant_name == "full":
            md_tmp = db_dir / "md"  # temp folder alongside the DB
            rag.preprocess(
                settings="d",  # convert PDFs + deduplicate
                input_dir=PDF_DIR,
                output_dir=md_tmp,
                preprocessor="md"
            )
            source_dir = md_tmp  # embed from the freshly created .md files
        else:
            source_dir = PDF_DIR

        # determine source_dir as you already do
        rag.embed(source_dir=str(source_dir))

        rag.load_vector_db(language=LANG)


        rag.vsm.store and rag.vsm.store._collection.count() or sys.exit("❌  Vector store is empty – aborting run")

        # iterate over dataset items and prompt
        for item in dataset.items:
            prompt_text = item.input.get('content', '')
            try:
                rag.prompt(prompt=prompt_text)
                time.sleep(20)
            except Exception as e:
                print(f"Error on item {item.id}: {e}")

# flush all traces to Langfuse\ nlf.flush()
print("Benchmarking complete; check Langfuse UI for metrics.")
lf.flush()
print("Benchmarking complete; check Langfuse UI for metrics.")
