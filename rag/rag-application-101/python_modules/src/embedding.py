import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EmbeddingManager:
    """Handles the document embeding generation using SentenceTransformer"""

    def __init__(self, embedding_model_name:str="all-MiniLM-L6-v2"):
    """
    Initialize the EmbeddingManager.

    Args:
        embedding_model_name (str): Name of the embedding model to use.
    """
        self.embedding_model_name = embedding_model_name
        self.model = None
        self._load_model()

    def _load_model(self):
    """
    Load the SentenceTransformer embedding model.
    """
        try:
            print(f"Loading SentenceTransformer model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            print(f"Loaded SentenceTransformer model successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def chunk_document(documents, chunk_size=1000, chunk_overlap=200) -> List[Any]:
        # chunk the documents into smaller chucks for better RAG performance
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            length_function=len,
            separators = ["\n", "\n\n", " ", ""])
        chunks = text_splitter.split_documents(documents)
        return chunks

  def generate_embeddings(self, texts:List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
      texts (List[str]): List of texts to generate embeddings for.

    Returns:
      np.ndarray: Array of embeddings.
    """
    if self.model is None:
      self._load_model()
    try:
      print(f"Generating embeddings for {len(texts)} texts...")
      embeddings = self.model.encode(texts, show_progress_bar=True)
      print(f"Successfully generated embeddings for {len(texts)} texts with shape: {embeddings.shape}")
      return embeddings
    except Exception as e:
      print(f"Error generating embeddings: {e}")
      raise


# Initialize the embedding manager
embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in all_pdf_document])
