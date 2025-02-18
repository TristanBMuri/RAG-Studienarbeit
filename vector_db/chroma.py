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


def load_embeddings(directory):
    """
    Load embeddings from .npy files in a given directory:
      - Change to the directory
      - Find .npy files
      - Load each file
      - Concatenate arrays
    """
    if not change_directory(directory):
        return None
    file_paths = get_npy_file_paths(directory)
    arrays = load_npy_files(file_paths)
    combined_array = combine_arrays(arrays)
    return combined_array


def initialize_vector_db_langchain(embeddings, collection_name="embeddings_collection"):
    """
    Initialize an in-memory vector DB using LangChain's Chroma integration
    with the updated community packages.
    """
    # Create an embedding function using HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create an in-memory Chroma vector store by setting persist_directory to None.
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=None
    )

    # Prepare dummy texts since we only have embeddings.
    num_vectors = embeddings.shape[0]
    ids = [f"vec_{i}" for i in range(num_vectors)]
    dummy_texts = [f"Document {i}" for i in range(num_vectors)]

    # Add the precomputed embeddings to the underlying collection.
    vectorstore._collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=[{"index": i} for i in range(num_vectors)],
        documents=dummy_texts
    )
    print(f"Added {num_vectors} embeddings to the collection using LangChain's Chroma.")
    return vectorstore


def query_vector_db_langchain(vectorstore, query, k=3):
    """
    Query the vector DB using LangChain's similarity_search interface.
    """
    results = vectorstore.similarity_search(query, k=k)
    print("Query results:", results)
    return results


if __name__ == '__main__':
    directory = "C:/Users/trist/PycharmProjects/RAG-Studienarbeit/src/embeddings/"
    combined_embeddings = load_embeddings(directory)

    if combined_embeddings is not None:
        # Initialize the in-memory vector DB using LangChain's Chroma.
        vectorstore = initialize_vector_db_langchain(combined_embeddings)
        # Example query
        query = "example query"
        results = query_vector_db_langchain(vectorstore, query)
