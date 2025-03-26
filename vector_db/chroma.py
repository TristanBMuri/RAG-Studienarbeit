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

def query_documents_by_max_chunk(vector_store, input_query, k=520, top_n=3):
    query_results = vector_store.similarity_search_with_score(input_query, k=k)
    grouped = {}
    for doc, score in query_results:
        file_name = doc.metadata.get("file_name", "unknown")
        if file_name not in grouped or score > grouped[file_name]["score"]:
            grouped[file_name] = {"doc": doc, "score": score}
    sorted_docs = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
    return [(item["doc"], item["score"]) for item in sorted_docs[:top_n]]

def embed_chunks_from_directory(input_directory):
    md_file_paths = get_md_file_paths(str(input_directory))
    if not md_file_paths:
        print(f"No markdown files found in {input_directory}.")
        return

    md_documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

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


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    md_directory = base_dir / "../src/md_versions/"
    test_embed, markd_documents = embed_from_directory(md_directory)
    if test_embed is not None:
        vectorstore = initialize_vector_db_langchain(test_embed, markd_documents)
        query = "example query"
        results = query_vector_db_langchain(vectorstore, query, 1)
