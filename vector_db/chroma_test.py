import numpy as np
import glob
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Global model for generating query embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

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


def initialize_vector_db(embeddings, collection_name="embeddings_collection"):
    """
    Initialize an in-memory vector DB using Chroma with the given embeddings.
    """
    # Create an in-memory client by not passing any persistent parameters.
    client = chromadb.Client()
    # Retrieve or create a collection
    collection = client.get_or_create_collection(name=collection_name)

    # Generate unique IDs for each vector
    num_vectors = embeddings.shape[0]
    ids = [f"vec_{i}" for i in range(num_vectors)]

    # Add embeddings to the collection (convert numpy array to list)
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=[{"index": i} for i in range(num_vectors)]
    )
    print(f"Added {num_vectors} embeddings to the collection.")
    return collection


def get_query_embedding(query):
    """Generate an embedding for the given query using SentenceTransformer."""
    return model.encode(query)


def query_vector_db(collection, query, n_results=3):
    """
    Query the vector DB using the provided query.
    """
    query_embedding = get_query_embedding(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    print("Query results:", results)
    return results


if __name__ == '__main__':
    directory = "C:/Users/trist/PycharmProjects/RAG-Studienarbeit/src/embeddings/"
    combined_embeddings = load_embeddings(directory)

    if combined_embeddings is not None:
        # Initialize the in-memory vector DB with the loaded embeddings
        collection = initialize_vector_db(combined_embeddings)
        # Example query
        query = "example query"
        results = query_vector_db(collection, query)
