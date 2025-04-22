import unittest
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import chroma module
sys.path.append(str(Path(__file__).parent.parent))
from vector_db.chroma import (
    change_directory, 
    get_npy_file_paths, 
    load_npy_files, 
    combine_arrays,
    load_embeddings,
    initialize_vector_db_langchain,
    query_vector_db_langchain,
    embed_from_directory,
    load_vector_db_langchain,
    query_documents_by_max_chunk,
    add_single_document
)
from langchain.docstore.document import Document


class TestChromaModule(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures, if any."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create sample arrays for testing
        self.sample_array1 = np.random.random((5, 10))
        self.sample_array2 = np.random.random((3, 10))
        
        # Create sample documents for testing
        self.sample_docs = [
            Document(page_content="This is a test document 1", 
                    metadata={"file_name": "test1.md", "path": "/test/test1.md", "language": "en"}),
            Document(page_content="This is a test document 2", 
                    metadata={"file_name": "test2.md", "path": "/test/test2.md", "language": "en"})
        ]
        
        # Save sample arrays to temp directory for testing
        np.save(self.temp_path / "test_array1.npy", self.sample_array1)
        np.save(self.temp_path / "test_array2.npy", self.sample_array2)

    def tearDown(self):
        """Tear down test fixtures, if any."""
        self.temp_dir.cleanup()

    def test_change_directory(self):
        """Test changing directory function."""
        # Test with valid directory
        self.assertTrue(change_directory(self.temp_path))
        
        # Test with invalid directory
        self.assertFalse(change_directory("/nonexistent/directory"))

    def test_get_npy_file_paths(self):
        """Test finding .npy files in a directory."""
        # Change to the temp directory
        os.chdir(self.temp_path)
        
        # Get the file paths
        file_paths = get_npy_file_paths(self.temp_path)
        
        # Verify that we found the expected number of files
        self.assertEqual(len(file_paths), 2)
        
        # Verify that the file paths contain our test files
        file_names = [os.path.basename(path) for path in file_paths]
        self.assertIn("test_array1.npy", file_names)
        self.assertIn("test_array2.npy", file_names)

    def test_load_npy_files(self):
        """Test loading .npy files."""
        # Get the file paths
        file_paths = [
            str(self.temp_path / "test_array1.npy"),
            str(self.temp_path / "test_array2.npy")
        ]
        
        # Load the arrays
        arrays = load_npy_files(file_paths)
        
        # Verify that we loaded the expected number of arrays
        self.assertEqual(len(arrays), 2)
        
        # Verify that the arrays have the expected shapes
        self.assertEqual(arrays[0].shape, (5, 10))
        self.assertEqual(arrays[1].shape, (3, 10))

    def test_combine_arrays(self):
        """Test combining arrays."""
        # Create some test arrays
        arrays = [
            np.ones((2, 3)),
            np.zeros((3, 3))
        ]
        
        # Combine the arrays
        combined = combine_arrays(arrays)
        
        # Verify the shape of the combined array
        self.assertEqual(combined.shape, (5, 3))
        
        # Test with empty list
        self.assertIsNone(combine_arrays([]))

    @patch('vector_db.chroma.change_directory')
    @patch('vector_db.chroma.get_npy_file_paths')
    @patch('vector_db.chroma.load_npy_files')
    @patch('vector_db.chroma.combine_arrays')
    def test_load_embeddings(self, mock_combine, mock_load, mock_get_paths, mock_change_dir):
        """Test loading embeddings from a directory."""
        # Configure mocks
        mock_change_dir.return_value = True
        mock_get_paths.return_value = ["path1.npy", "path2.npy"]
        mock_load.return_value = [np.ones((2, 3)), np.zeros((3, 3))]
        mock_combine.return_value = np.ones((5, 3))
        
        # Call the function
        result = load_embeddings("/test/directory")
        
        # Verify the function was called with the correct arguments
        mock_change_dir.assert_called_once_with("/test/directory")
        mock_get_paths.assert_called_once()
        mock_load.assert_called_once()
        mock_combine.assert_called_once()
        
        # Verify the result
        self.assertEqual(result.shape, (5, 3))
        
        # Test with directory change failure
        mock_change_dir.return_value = False
        result = load_embeddings("/test/directory")
        self.assertIsNone(result)

    @patch('langchain_huggingface.HuggingFaceEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_initialize_vector_db_langchain(self, mock_chroma, mock_embeddings):
        """Test initializing vector DB with LangChain."""
        # Configure mocks
        mock_embeddings.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Create test data
        embeddings = np.ones((10, 128))
        documents = self.sample_docs
        
        # Call the function
        result = initialize_vector_db_langchain(embeddings, documents)
        
        # Verify the function was called with the correct arguments
        mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        mock_chroma.assert_called_once()
        
        # Verify that add was called on the collection
        self.assertTrue(mock_chroma_instance._collection.add.called)
        
        # Verify the result
        self.assertEqual(result, mock_chroma_instance)

    @patch('langchain_huggingface.HuggingFaceEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_query_vector_db_langchain(self, mock_chroma, mock_embeddings):
        """Test querying vector DB with LangChain."""
        # Configure mocks
        mock_chroma_instance = MagicMock()
        mock_chroma_instance.similarity_search_with_score.return_value = [
            (Document(page_content="Test content"), 0.95)
        ]
        
        # Call the function
        results = query_vector_db_langchain(mock_chroma_instance, "test query", 3)
        
        # Verify the function was called with the correct arguments
        mock_chroma_instance.similarity_search_with_score.assert_called_once_with("test query", k=3)
        
        # Verify the results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].page_content, "Test content")
        self.assertEqual(results[0][1], 0.95)

    @patch('langchain_huggingface.HuggingFaceEmbeddings')
    def test_embed_from_directory(self, mock_embeddings):
        """Test embedding documents from a directory."""
        # Configure mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Create test files
        (self.temp_path / "test1.md").write_text("This is test document 1")
        (self.temp_path / "test2_en.md").write_text("This is test document 2")
        
        # Call the function
        embeddings, documents = embed_from_directory(self.temp_path)
        
        # Verify the embeddings
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(documents), 2)
        
        # Verify document metadata
        self.assertIn("file_name", documents[0].metadata)
        self.assertIn("path", documents[0].metadata)
        self.assertIn("language", documents[0].metadata)

    @patch('langchain_huggingface.HuggingFaceEmbeddings')
    @patch('langchain_chroma.Chroma')
    def test_load_vector_db_langchain(self, mock_chroma, mock_embeddings):
        """Test loading a vector DB with LangChain."""
        # Configure mocks
        mock_embeddings.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance
        
        # Call the function
        result = load_vector_db_langchain()
        
        # Verify the function was called with the correct arguments
        mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        mock_chroma.assert_called_once()
        
        # Verify the result
        self.assertEqual(result, mock_chroma_instance)

    @patch('langchain_huggingface.HuggingFaceEmbeddings')
    def test_add_single_document(self, mock_embeddings):
        """Test adding a single document to the vector store."""
        # Configure mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2]]
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Create mock vector store
        mock_vector_store = MagicMock()
        
        # Create test document
        doc = Document(page_content="Test document", metadata={"source": "test"})
        
        # Call the function
        add_single_document(mock_vector_store, doc)
        
        # Verify the collection.add was called
        mock_vector_store._collection.add.assert_called_once()


if __name__ == '__main__':
    unittest.main()
