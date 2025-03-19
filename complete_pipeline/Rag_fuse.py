import getpass

from langfuse import Langfuse

from complete_pipeline.prompt.langfuseprompter import LangfusePrompter
from complete_pipeline.prompt.query_result import QueryResult
from preprocessing_scripts.mark_any_down import mark_any_down
from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown
from preprocessing_scripts.any_unstructured import *
from vector_db.chroma import *
import uuid
import os

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
    def __init__(self, custom_handler=None, prompt_template=None):
        self.model = None  # Placeholder for an LLM model
        self.md_folder = Path("../src/md_versions")
        self.vectorstore = None
        if prompt_template is None:
            self.prompt_template  = "rag_prompt"
            self.chat = LangfusePrompter(custom_handler=custom_handler)
        else:
            self.chat = LangfusePrompter(prompt_name=prompt_template, custom_handler=custom_handler)
            self.prompt_template = prompt_template
        self.sessionId = _generate_session_id()
        self.langfuse = Langfuse()

    def preprocess(self, settings = "d"):
        # Prompt user for input and output directories
        if not settings:
            settings = input("Set the preprocessing settings (c for custom, d for default and skip to skip): ")

        if settings == "skip" or settings == "s":
            self.md_folder = Path("../src/md_versions")
            return None

        if settings == "d":
            input_dir = Path("../Docs")
            output_dir = Path("../src/md_versions")
            _md_preprocess(input_dir, output_dir)
            self.md_folder = output_dir
            return None

        if settings == "c":
            preprocessor_choice = input("Type md for markitdown preprocessing or leave empty for default: ")
            input_dir = input("Enter the document folder directory (or type N for default): ")
            output_dir = input("Enter the output folder directory (or type N for default): ")

            if preprocessor_choice == "u":
                if input_dir.lower() == "n":
                    input_dir = Path("../Docs")
                else:
                    input_dir = Path(input_dir)

                if output_dir.lower() == "n":
                    output_dir = Path("../src/Unstructured")
                else:
                    output_dir = Path(output_dir)

                self.md_folder = output_dir
                convert_folder(input_dir, output_dir)


            if preprocessor_choice == "md":
                if input_dir.lower() == "n":
                    input_dir = Path("../Docs")
                else:
                    input_dir = Path(input_dir)

                if output_dir.lower() == "n":
                    output_dir = Path("../src/md_versions")
                else:
                    output_dir = Path(output_dir)

                _md_preprocess(input_dir, output_dir)
                self.md_folder = output_dir

    def embed(self):
        embeddings_array, md_documents = embed_from_directory(self.md_folder)

        # Hier übergeben wir sowohl embeddings_array als auch documents
        self.vectorstore = initialize_vector_db_langchain(embeddings_array, md_documents)
        print("Vector store initialized.")

    def query(self):
        """
        Prompts for a search query and returns the n most relevant documents from the vector DB.
        """
        if self.vectorstore is None:
            print("Vector store not initialized. Please run embed() first.")
            return []
        query_input = input("Enter your search query: ")
        query_results = query_vector_db_langchain(self.vectorstore, query_input, k=3)
        return query_results

    def prompt(self, prompt=None, custom_handler=None):
        """
        Takes a user question, retrieves context from the vector DB,
        and simulates sending it to an LLM to generate a response.
        """
        if prompt is None:
            query_input = input("Enter your question: ")
        else:
            query_input = prompt

        if custom_handler is None:
            trace = self.langfuse.trace(name="RAG Query", input={"question": query_input}, tags=["RAG"])
        else:
            trace = custom_handler.trace

        # Todo change this
        if self.vectorstore is None:
            print("Vector store not initialized. Please run embed() first.")
            return

        retrieval = trace.span(name="Document Retrieval", input={"query": query_input})
        search_span = retrieval.span(name="Query Search")
        # Retrieve results as (Document, score) tuples.
        query_results = query_vector_db_langchain(
            self.vectorstore,
            query_input,
            k=3,
            session_id=self.sessionId
        )
        search_span.end(output={"documents_found": len(query_results)})

        text_extraction_span = retrieval.span(name="Text Re-Extraction")
        # Extract the context from the page_content of each Document.
        documents = [doc for doc, score in query_results]
        context = "\n".join([doc.page_content for doc in documents])

        # Build a metadata summary string for each result.
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
            prompt= self.langfuse.get_prompt(self.prompt_template),
            input={"query": query_input, "docs": documents})

        # Build the QueryResult with the metadata context and the document context.
        qr_response = QueryResult(meta_context, context)
        response = self.chat.prompt(
            query_input, qr_response, session_id=self.sessionId, full_response=True)

        usage_metadata = getattr(response, 'usage_metadata', {})
        usage_details = {
            "input": usage_metadata.get("input_tokens", 0),
            "output": usage_metadata.get("output_tokens", 0),
            # While interesting langfuse just counts the token amount together which isn't a true amount
            #"token_usage": usage_metadata.get("total_tokens", 0),
            #"prompt_tokens": usage_metadata.get("input_tokens", 0),
            #"completion_tokens": usage_metadata.get("output_tokens", 0),
            #"total_tokens": usage_metadata.get("total_tokens", 0),
            #"unit": 1  # or another valid integer value if required
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

    # Run preprocessing to generate markdown versions of documents
    rag.preprocess("s")

    # Compute embeddings and initialize the vector DB
    rag.embed()

    # test_question = " In welchen sprachen ist Easygest verfügbar? Und wo werden die Daten gespeichert? "
    test_question = " Welche Versicherungen beteiligen sich an der Finanzierung von Hilfsmitteln und Krankentransporten? "
    re = rag.prompt(test_question)
    print(re)
    # Interactive loop for querying and prompting
    while True:
        break
        action = input("Enter action (query, prompt, or exit): ").lower()
        if action == "query":
            results = rag.query()
            print("Query results:", results)
        elif action == "prompt":
            print(rag.prompt())
        elif action == "exit":
            break
        else:
            print("Invalid action. Please choose 'query', 'prompt', or 'exit'.")
