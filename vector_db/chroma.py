from pathlib import Path

import numpy as np
import glob
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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
    Load embeddings from .npy files in a given directory:
      - Change to the directory
      - Find .npy files
      - Load each file
      - Concatenate arrays
    """
    if not change_directory(embeddings_directory):
        return None
    file_paths = get_npy_file_paths(embeddings_directory)
    arrays = load_npy_files(file_paths)
    combined_array = combine_arrays(arrays)
    return combined_array


def initialize_vector_db_langchain(embeddings, input_documents, collection_name="embeddings_collection"):
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=None
    )

    num_vectors = embeddings.shape[0]
    ids = [f"vec_{i}" for i in range(num_vectors)]

    vector_store._collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=[{"index": i} for i in range(num_vectors)],
        documents=input_documents
    )
    print(f"Added {num_vectors} embeddings to the collection using LangChain's Chroma.")
    return vector_store



def query_vector_db_langchain(vector_store, input_query, k=3):
    """
    Query the vector DB using LangChain's similarity_search interface.
    """
    # TO consider: Thresholding for relevance and up maximum amount of documents to return
    query_results = vector_store.similarity_search_with_score(input_query, k=k)
    print("Query results:", query_results)
    return query_results

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
                md_documents.append(md_content)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Filter out invalid documents
    md_documents = [doc for doc in md_documents if doc]

    print(f"Embedding {len(md_documents)} documents...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_function.embed_documents(md_documents)
    embeddings_array = np.array(embeddings)
    return embeddings_array, md_documents



if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    md_directory = base_dir / "../src/md_versions/"

    test_embed, markd_documents = embed_from_directory(md_directory)

    if test_embed is not None:
        # Initialize the in-memory vector DB using LangChain's Chroma.
        vectorstore = initialize_vector_db_langchain(test_embed, markd_documents)
        # Example query
        query = "example query"
        results = query_vector_db_langchain(vectorstore, query, 1)
