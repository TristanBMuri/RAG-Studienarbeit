import getpass

from chromadb import Settings
from langfuse import Langfuse
from complete_pipeline.prompt.langfuseprompter import LangfusePrompter
from complete_pipeline.prompt.query_result import QueryResult
from preprocessing_scripts.mark_any_down import mark_any_down
from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown
from preprocessing_scripts.any_unstructured import *
from vector_db.chroma import *
import uuid
import os
from pathlib import Path

def _generate_session_id():
    return uuid.uuid4().hex

def _md_preprocess(input_dir, output_dir):
    md_folder = Path(output_dir)
    md_folder.mkdir(parents=True, exist_ok=True)
    # Conversion of PDFs
    convert_pdfs_to_markdown(input_dir, output_dir)
    # Conversion of non-PDFs
    for each_file in input_dir.iterdir():
        if each_file.suffix.lower() == ".pdf":
            continue
        else:
            mark_any_down(each_file, output_dir)

class RagFuse:
    def __init__(self, custom_handler=None, prompt_template=None, vector_store_dir=None, language="de"):
        self.model = None  # Placeholder for an LLM model
        self.md_folder = Path("../src/md_versions")
        if vector_store_dir is None:
            self.vectorstore = None
        else:
            self.vectorstore = None
            self.load_vector_db(language=language, persist_directory=vector_store_dir)
        if prompt_template is None:
            self.prompt_template  = "rag_prompt"
            self.chat = LangfusePrompter(custom_handler=custom_handler)
        else:
            self.chat = LangfusePrompter(prompt_name=prompt_template, custom_handler=custom_handler)
            self.prompt_template = prompt_template
        self.sessionId = _generate_session_id()
        self.langfuse = Langfuse()

    def preprocess(self, settings="d", input_dir=Path("../Docs"), output_dir=Path("../src/md_versions"), preprocessor="md"):
        if settings == "skip" or settings == "s":
            self.md_folder = Path("../src/md_versions")
            return None
        if preprocessor == "md":
            _md_preprocess(input_dir, output_dir)
            self.md_folder = output_dir
            return None
        if preprocessor == "u":
            # unstructured default is currently Path("../src/Unstructured")
            self.md_folder = output_dir
            convert_folder(input_dir, output_dir)

    def embed(self, language, source_dir=None):
        """
        Embed documents for a given language.
        Only documents whose file name indicates the target language (or those without a language tag, assumed English)
        are included in the vector DB.
        """
        if source_dir:
            self.md_folder = Path(source_dir)
        embeddings_array, md_documents = embed_from_directory(self.md_folder)
        # Filter: include documents whose language is the target language or "en"
        indices = [i for i, doc in enumerate(md_documents) if doc.metadata.get("language") in [language, "en"]]
        if not indices:
            print(f"No documents found for language {language}.")
            return
        filtered_embeddings = embeddings_array[indices]
        filtered_docs = [md_documents[i] for i in indices]
        collection_name = f"embeddings_collection_{language}"
        persist_directory = f"./chroma_db_{language}"
        self.vectorstore = initialize_vector_db_langchain(filtered_embeddings, filtered_docs,
                                                          collection_name=collection_name,
                                                          persist_directory=persist_directory)
        print("Vector store initialized for language:", language)

    def load_from_embedding(self, embeddings_dir: Path):
        self.vectorstore = load_embeddings(embeddings_dir)

    def load_vector_db(self, language, persist_directory=None):
        collection_name = f"embeddings_collection_{language}"
        if persist_directory is None:
            persist_directory = f"./chroma_db_{language}"

        # This assumes load_vector_db_langchain is a helper function in chroma.py that loads an existing collection.
        self.vectorstore = load_vector_db_langchain(collection_name=collection_name,
                                                    persist_directory=persist_directory)
        print(f"Loaded vector DB for language {language} from {persist_directory}")

    def query(self):
        if self.vectorstore is None:
            print("Vector store not initialized. Please run embed() first.")
            return []
        query_input = input("Enter your search query: ")
        query_results = query_vector_db_langchain(self.vectorstore, query_input, k=3)
        return query_results

    def prompt(self, prompt=None, custom_handler=None):
        if prompt is None:
            query_input = input("Enter your question: ")
        else:
            query_input = prompt

        if custom_handler is None:
            trace = self.langfuse.trace(name="RAG Query", input={"question": query_input}, tags=["RAG"])
        else:
            trace = custom_handler.trace

        if self.vectorstore is None:
            print("Vector store not initialized. Please run embed() first.")
            return

        retrieval = trace.span(name="Document Retrieval", input={"query": query_input})
        search_span = retrieval.span(name="Query Search")
        query_results = query_vector_db_langchain(self.vectorstore, query_input, k=3)
        search_span.end(output={"documents_found": len(query_results)})

        text_extraction_span = retrieval.span(name="Text Re-Extraction")
        documents = [doc for doc, score in query_results]
        context = "\n".join([doc.page_content for doc in documents])

        meta_context_list = []
        for doc, score in query_results:
            meta = doc.metadata.copy()
            meta.pop("page_content", None)
            meta["score"] = score
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
            meta_context_list.append(meta_str)
        meta_context = "\n".join(meta_context_list)

        text_extraction_span.end(output={"documents": documents})
        retrieval.end(output={"results_count": len(documents)})

        answer_gen = trace.generation(
            name="Answer Generation",
            model=self.chat.get_model(),
            prompt=self.langfuse.get_prompt(self.prompt_template),
            input={"query": query_input, "docs": documents})

        qr_response = QueryResult(meta_context, context)
        response = self.chat.prompt(query_input, qr_response, session_id=self.sessionId, full_response=True)

        usage_metadata = getattr(response, 'usage_metadata', {})
        usage_details = {
            "input": usage_metadata.get("input_tokens", 0),
            "output": usage_metadata.get("output_tokens", 0),
        }

        answer_gen.end(output={"answer": response}, usage_details=usage_details)
        trace.update(output={"answer": getattr(response, "content", response)},
                     usage_details={"token_usage": getattr(response, 'usage', {})})
        self.langfuse.flush()
        return response

if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    rag = RagFuse(prompt_template="combined_rag")
    language = "de"
    try:
        rag.load_vector_db(language=language)
    except Exception as e:
        print(e)
    client = rag.vectorstore._client
    collection = client.get_collection(f"embeddings_collection_{language}")

    if collection.count() == 0:
        output_dir = Path("../src/company_pdfs")
        #rag.preprocess(input_dir=Path("./ethical_crawler/company_pdfs"), output_dir=output_dir, preprocessor="u")
        rag.embed(language="de", source_dir=output_dir)
    # Example: Initialize for German language chatbot
    test_question = "Welche Versicherungen beteiligen sich an der Finanzierung von Hilfsmitteln und Krankentransporten?"
    re = rag.prompt(test_question)
    print(re)
