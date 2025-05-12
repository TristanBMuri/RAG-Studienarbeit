import json
from pathlib import Path
import uuid
from typing import List

from langfuse import Langfuse
from numpy.f2py.auxfuncs import throw_error
import re

from complete_pipeline.prompt.langfuseprompter import LangfusePrompter
from complete_pipeline.prompt.query_result import QueryResult
from preprocessing_scripts.mark_any_down import mark_any_down
from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown
from preprocessing_scripts.any_unstructured import convert_folder
from langchain.docstore.document import Document
from vector_db.vector_store_manager import VectorStoreManager, VectorStoreConfig


def _generate_session_id() -> str:
    return uuid.uuid4().hex


def _md_preprocess(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    convert_pdfs_to_markdown(input_dir, output_dir)
    for each_file in input_dir.iterdir():
        if each_file.suffix.lower() != ".pdf":
            mark_any_down(each_file, output_dir)


class RagFuse:
    def __init__(
        self,
        custom_handler=None,
        prompt_template: str = None,
        vector_store_dir: str = None,
        language: str = "de"
    ):
        self.language = language
        self.model = None  # placeholder for LLM
        self.md_folder = Path("../src/md_versions")
        # Langfuse setup
        if prompt_template:
            self.prompt_template = prompt_template
            self.chat = LangfusePrompter(prompt_name=prompt_template, custom_handler=custom_handler)
        else:
            self.prompt_template = "rag_prompt"
            self.chat = LangfusePrompter(custom_handler=custom_handler)
        self.sessionId = _generate_session_id()
        self.langfuse = Langfuse()
        # Vector store manager
        if vector_store_dir:
            cfg = VectorStoreConfig(
                collection_name=f"embeddings_collection_{language}",
                persist_directory=vector_store_dir
            )
            self.vsm = VectorStoreManager(cfg)
            self.vsm.load_store()
        else:
            self.vsm = None

    def preprocess(
        self,
        settings: str = "d",
        input_dir: Path = Path("../Docs"),
        output_dir: Path = Path("../src/md_versions"),
        preprocessor: str = "md"
    ) -> None:
        if settings in ("skip", "s"):
            self.md_folder = Path("../src/md_versions")
            return
        if preprocessor == "md":
            _md_preprocess(input_dir, output_dir)
            self.md_folder = output_dir
        else:
            convert_folder(input_dir, output_dir)
            self.md_folder = output_dir

    def embed(
        self,
        source_dir: Path = None
    ) -> None:
        """
        Embed all markdown docs using VectorStoreManager.
        """
        md_dir = Path(source_dir) if source_dir else self.md_folder
        # read markdown files
        docs = self._md_glob(md_dir)
        embeddings, filtered = self.vsm.embed_documents(docs)  # full-doc embeddings
        self.vsm.initialize_store(embeddings, filtered)

    def _md_glob(self, md_dir) -> (List[Document]):
        docs = []
        for f in md_dir.glob("*.md"):
            content = f.read_text(encoding="utf-8")
            # infer language metadata
            lang = 'en'
            if '_de' in f.name:
                lang = 'de'
            elif '_fr' in f.name:
                lang = 'fr'
            docs.append(Document(page_content=content, metadata={"file_name": f.name, "path": str(f), "language": lang}))
        return docs

    def load_vector_db(self, language: str, persist_directory: str = None) -> None:
        if not self.vsm:
            cfg = VectorStoreConfig(
                collection_name=f"embeddings_collection_{language}",
                persist_directory=persist_directory or f"./chroma_db_{language}"
            )
            self.vsm = VectorStoreManager(cfg)
        self.vsm.load_store()

    def prompt(
        self,
        prompt: str = None,
        custom_handler=None,
        k: int = 3,
        top_n: int = 3,
        full_doc: bool = False
    ):
        query_input = prompt
        if not query_input:
            throw_error("Please provide a query.")
        # start trace
        trace = (
            custom_handler.trace
            if custom_handler
            else self.langfuse.trace(
                name="RAG Query",
                input={"question": query_input},
                tags=["RAG", f"embed:{self.vsm.config.embedding_model}"]
            )
        )

        # ensure store
        if not self.vsm or not self.vsm.store:
            print("Vector store not initialized. Please run embed() or load_vector_db() first.")
            return
        # document retrieval
        retrieval = trace.span(name="Document Retrieval", input={"query": query_input})
        search_span = retrieval.span(name="Query Search")
        query_results = self.vsm.query(query_input, k=k)

        doc_names = [doc.metadata.get("file_name", "") for doc, _ in query_results]
        top_n_doc_names = []
        for doc_name in doc_names:
            if doc_name not in top_n_doc_names:
                top_n_doc_names.append(doc_name)
            if len(top_n_doc_names) >= top_n:
                break
        # todo only take 3 not all chunks for the 3 documents
        top_n_docs = []
        for doc, _ in query_results:
            if doc.metadata.get("file_name", "") in top_n_doc_names:
                top_n_docs.append(doc)
                top_n_doc_names.remove(doc.metadata.get("file_name", ""))
                if len(top_n_docs) >= top_n:
                    break

        search_span.end(output={
            "documents_found": len(query_results),
            "document_names": doc_names
        })
        # build context
        if not full_doc:
            documents = [doc for doc, score in query_results]
        else:
            documents = []
            doc_names = [doc.metadata.get("file_name", "") for doc in top_n_docs]

            for doc in top_n_docs:
                file_name = doc.metadata.get("file_name")
                file_path = self.md_folder / file_name
                if not file_path.exists():
                    raise FileNotFoundError(f"Expected file {file_path} not found")
                full_text = file_path.read_text(encoding="utf-8")
                documents.append(Document(page_content=full_text, metadata=doc.metadata))
            
        context = "\n".join([d.page_content for d in documents])
        meta_context = "\n".join([
            ", ".join(f"{k}: {v}" for k, v in doc.metadata.items()) + f", score: {score}"
            for doc, score in query_results
        ])
        retrieval.end(output={"results_count": len(documents), "document_names": doc_names})

        # answer generation
        answer_gen = trace.generation(
            name="Answer Generation",
            model=self.chat.get_model(),
            prompt=self.langfuse.get_prompt(self.prompt_template),
            input={"query": query_input, "docs": documents}
        )
        qr_response = QueryResult(meta_context, context)
        response = self.chat.prompt(query_input, qr_response, session_id=self.sessionId, full_response=True)
        # corrected usage details
        usage = getattr(response, 'usage_metadata', {})
        prompt_tokens = usage.get('input_tokens', 0)
        completion_tokens = usage.get('output_tokens', 0)
        total_tokens = prompt_tokens + completion_tokens

        try:
            record_categorical_scores(trace, response)
        except Exception as exc:
            trace.span(name="Score Recording Error").end(output={"error": str(exc)})

        answer_gen.end(
            output={"answer": getattr(response, 'content', response)},
            usage_details={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        trace.update(
            output={"answer": getattr(response, 'content', response)},
            usage_details={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        return response


def extract_first_json(text: str) -> dict:
    """Return the *first* JSON object found in *text* (raises if none)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(match.group(0))


def record_categorical_scores(trace, response) -> dict:
    """Parse *response*, store categorical scores on *trace*, return the values.

    Parameters
    ----------
    trace : langfuse.Trace
        The trace previously opened for this LLM call.
    response : langfuse.Generation | str
        The raw LLM response (Generation or plain string).

    Returns
    -------
    dict
        A dict with the stored field→value pairs (useful for debugging).
    """
    raw_text = getattr(response, "content", str(response))
    data = extract_first_json(raw_text)

    stored = {}
    for field in ("additional_context_eval", "context_quality_eval"):
        if field not in data:
            raise KeyError(f"Field '{field}' missing in response JSON")
        value = data[field]
        # Write directly as categorical
        Langfuse().score(
            trace_id=trace.id,
            name=field,
            value=value,
            data_type="CATEGORICAL",
            comment=value,
        )
        stored[field] = value
    return stored
