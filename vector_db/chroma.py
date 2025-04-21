import uuid
from pathlib import Path
import numpy as np
import glob
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langfuse.decorators import observe, langfuse_context

def change_directory(directory):
    """Change to the target directory and handle errors."""
    try:
        os.chdir(directory)
        return True
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return False

def get_npy_file_paths(directory):
    """Retrieve all .npy file paths from the given directory."""
    file_paths = glob.glob(os.path.join(directory, '*.npy'))
    print(f"Found {len(file_paths)} .npy files.")
    return file_paths

def load_npy_files(file_paths):
    """Load each .npy file and return a list of non-empty arrays."""
    arrays = []
    for file in file_paths:
        try:
            array = np.load(file)
            if array.size > 0:
                arrays.append(array)
                print(f"Loaded {file} with shape {array.shape}.")
            else:
                print(f"File {file} is empty.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    print(f"Total arrays loaded: {len(arrays)}")
    return arrays

def combine_arrays(arrays):
    """Concatenate all arrays along axis=0."""
    if arrays:
        return np.concatenate(arrays, axis=0)
    else:
        print("No arrays to concatenate.")
        return None

def load_embeddings(embeddings_directory):
    """
    Load embeddings from .npy files in a given directory.
    """
    if not change_directory(embeddings_directory):
        return None
    file_paths = get_npy_file_paths(embeddings_directory)
    arrays = load_npy_files(file_paths)
    combined_array = combine_arrays(arrays)
    return combined_array

def initialize_vector_db_langchain(embeddings, documents, collection_name="embeddings_collection", persist_directory="./chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    num_vectors = embeddings.shape[0]
    ids = [f"vec_{i}" for i in range(num_vectors)]

    batch_size = 160  # must be <= 166
    for i in range(0, len(documents), batch_size):
        batch_ids = [f"vec_{j}" for j in range(i, min(i + batch_size, len(documents)))]
        batch_embeddings = embeddings[i:i + batch_size].tolist()
        batch_metadatas = [doc.metadata for doc in documents[i:i + batch_size]]
        batch_texts = [doc.page_content for doc in documents[i:i + batch_size]]

        vector_store._collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_texts
        )
    print(f"Added {num_vectors} embeddings to the collection '{collection_name}' in '{persist_directory}'.")
    return vector_store

def initialize_chunked_vector_db_langchain(embeddings, markd_documents, collection_name="embeddings_collection", persist_directory="./chroma_db"):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    num_vectors = embeddings.shape[0]
    batch_size = 160  # must be <= 166
    for i in range(0, len(markd_documents), batch_size):
        batch_ids = [f"vec_{j}" for j in range(i, min(i + batch_size, len(markd_documents)))]
        batch_embeddings = embeddings[i:i + batch_size].tolist()
        batch_metadatas = [doc.metadata for doc in markd_documents[i:i + batch_size]]
        batch_texts = [doc.page_content for doc in markd_documents[i:i + batch_size]]

        vector_store._collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_texts
        )

    print(f"Added {num_vectors} embeddings to the collection '{collection_name}' in '{persist_directory}'.")
    return vector_store

def query_vector_db_langchain(vector_store, input_query, k=3):
    query_results = vector_store.similarity_search_with_score(input_query, k=k)
    print("Query results:", query_results)
    return query_results

def query_documents_by_max_chunk(vector_store, input_query, k=50, top_n=3):
    query_results = vector_store.similarity_search_with_score(input_query, k=k)
    grouped = {}
    for doc, score in query_results:
        file_name = doc.metadata.get("file_name", "unknown")
        if file_name not in grouped or score > grouped[file_name]["score"]:
            grouped[file_name] = {"doc": doc, "score": score}
    sorted_docs = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
    return [(item["doc"], item["score"]) for item in sorted_docs[:top_n]]

def embed_chunks_from_directory(input_directory, chunk_size=300, chunk_overlap=50):
    md_file_paths = get_md_file_paths(str(input_directory))
    if not md_file_paths:
        print(f"No markdown files found in {input_directory}.")
        return

    md_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for file_path in md_file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                md_content = file.read()
            chunks = text_splitter.split_text(md_content)
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "file_name": Path(file_path).name,
                        "chunk_index": idx,
                        "path": file_path
                    }
                )
                md_documents.append(doc)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Created {len(md_documents)} document chunks from {len(md_file_paths)} files.")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_function.embed_documents([doc.page_content for doc in md_documents])
    embeddings_array = np.array(embeddings)
    return embeddings_array, md_documents

def get_md_file_paths(directory):
    file_paths = glob.glob(os.path.join(directory, '*.md'))
    print(f"Found {len(file_paths)} markdown files.")
    return file_paths

def embed_from_directory(input_directory):
    md_files = list(input_directory.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_directory}.")
        return

    md_documents = []
    for f in md_files:
        try:
            with open(f, "r", encoding="utf-8") as b:
                md_content = b.read()
                file_name = f.name
                if "_de" in file_name:
                    lang = "de"
                elif "_fr" in file_name:
                    lang = "fr"
                elif "_it" in file_name:
                    lang = "it"
                else:
                    lang = "en"
                doc = Document(
                    page_content=md_content,
                    metadata={"file_name": file_name, "path": str(f), "language": lang}
                )
                md_documents.append(doc)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Embedding {len(md_documents)} documents...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_function.embed_documents([doc.page_content for doc in md_documents])
    embeddings_array = np.array(embeddings)
    return embeddings_array, md_documents

def load_vector_db_langchain(collection_name="embeddings_collection_de", persist_directory="./chroma_db_de"):
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )
    return vector_store


import os
import shutil
import uuid
from pathlib import Path

from preprocessing_scripts.pdf_to_md_converter import convert_pdfs_to_markdown
from preprocessing_scripts.mark_any_down import mark_any_down
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings


def add_single_document(vector_store, document, embedding_function=None):
    """
    Adds a single document to the vector database.
    """
    if embedding_function is None:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Compute the embedding for the document text.
    embedding = embedding_function.embed_documents([document.page_content])[0]

    # Create a unique ID.
    doc_id = f"vec_{uuid.uuid4().hex}"

    # Add the document to the vector store.
    vector_store._collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[document.metadata],
        documents=[document.page_content]
    )
    print(f"Document added with ID: {doc_id}")


def process_and_add_file(file_path, vector_store, temp_md_dir=Path("./temp_md")):
    """
    Processes a single file:
      - If the file is a PDF, converts it to Markdown using
        convert_pdfs_to_markdown.
      - Otherwise, converts the file using mark_any_down.
    The resulting Markdown content is then used to create a Document,
    compute its embedding, and add it to the vector database.

    :param file_path: Path to the input file.
    :param vector_store: A pre-initialized vector store instance.
    :param temp_md_dir: Directory where Markdown files will be saved.
    """
    file_path = Path(file_path)
    temp_md_dir.mkdir(parents=True, exist_ok=True)

    # Prepare to store the Markdown content.
    if file_path.suffix.lower() == ".pdf":
        # Create a temporary directory to mimic the expected input to convert_pdfs_to_markdown.
        temp_input_dir = file_path.parent / "temp_input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)

        # Copy the PDF to the temporary input directory.
        dest_file = temp_input_dir / file_path.name
        shutil.copy(file_path, dest_file)

        # Convert the PDF to Markdown.
        convert_pdfs_to_markdown(temp_input_dir, temp_md_dir)

        # Assume the output Markdown file is named as the input file with a .md extension.
        md_file = temp_md_dir / (file_path.stem + ".md")

        # Clean up the temporary input directory.
        shutil.rmtree(temp_input_dir)
    else:
        # For non-PDFs, call mark_any_down to convert the file.
        mark_any_down(file_path, temp_md_dir)
        md_file = temp_md_dir / (file_path.stem + ".md")

    # Ensure the Markdown file was created.
    if not md_file.exists():
        print(f"Markdown file not found: {md_file}")
        return

    # Read the Markdown content.
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create a document from the Markdown.
    doc = Document(
        page_content=content,
        metadata={"source": file_path.name, "original_path": str(file_path)}
    )

    # Add the document to the vector database.
    add_single_document(vector_store, doc)

    # Optionally, clean up the temporary Markdown file.
    # md_file.unlink()


# Example usage:
if __name__ == "__main__":
    # Assume vector_store is already initialized (e.g. via load_vector_db_langchain)
    vector_store = load_vector_db_langchain(collection_name="embeddings_collection_en",
                                            persist_directory="./chroma_db_en")

    # Path to your input file (could be a PDF or another file type)
    input_file = "path/to/your/document.pdf"  # or a .txt, .docx, etc.

    process_and_add_file(input_file, vector_store)

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    md_directory = base_dir / "../src/md_versions/"
    test_embed, markd_documents = embed_from_directory(md_directory)
    if test_embed is not None:
        vectorstore = initialize_vector_db_langchain(test_embed, markd_documents)
        query = "example query"
        results = query_vector_db_langchain(vectorstore, query, 1)
