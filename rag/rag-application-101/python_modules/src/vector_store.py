from pathlib import PureWindowsPath
from typing import Self, Any
import os
import chromadb
import numpy as np

class ChromaVectorStore:
  """ Manages document embeddings into a ChromaDB vector store"""
  def __init__(self, collection_name="pdf_documents", persist_directory="./data/vector_store"):
    """
    Initialize the vector store

    Arg:
      collection_name: Name of ChromaDB collection
      persist_directory: Path to store ChromaDB data
    """
    self.collection_name = collection_name
    self.persist_directory = persist_directory
    self._initialize_store()

  def _initialize_store(self):
    """ Initialize the ChromaDB client and collection"""
    try:
      # Create persistent ChromaDB client
      os.makedirs(self.persist_directory, exist_ok=True)
      self.client = chromadb.PersistentClient(path=self.persist_directory)

      # Get or create collection
      self.collection = self.client.get_or_create_collection(
          name=self.collection_name,
          metadata={"description":"PDF document embeddings for RAG"})

      print(f"ChromaDB Vector Store initialized with collection: '{self.collection_name}' at path: '{self.persist_directory}'")
      print(f"Existing documents in the collection: {self.collection.count()}")

    except Exception as e:
      print(f"Error initializing ChromaDB vector store db: {e}")
      raise

  def add_documents(self, documents: list[Any], embeddings: np.ndarray):
    """ Add documents and their embeddings to the vector store

    Args:
      documents: List of documents to add
      embeddings: List of embeddings for the documents
    """
    if len(documents) != len(embeddings):
      raise ValueError("Number of documents and embeddings must be the same")

    print(f"Adding {len(documents)} documents to the vector store")

    # Prepare data for Chroma DB
    ids = []
    metadatas = []
    documents_text = []
    embeddings_list = []

    for i, (document, embedding) in enumerate(zip(documents, embeddings)):
      # generate unique Id
      document_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
      ids.append(document_id)

      # prepare metadata
      metadata = dict(document.metadata)
      metadata['doc_index'] = i
      metadata['content_length'] = len(document.page_content)
      metadatas.append(metadata)

      # prepare document text
      documents_text.append(document.page_content)

      # Embedding
      embeddings_list.append(embedding.tolist())

    # Add data to Chroma DB
    self.collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=documents_text,
        embeddings=embeddings_list
    )
    print(f"Added {len(documents)} documents to the vector store")
    print(f"Total documents in the collection: {self.collection.count()}")
    