from src.data_loader import load_all_document
from src.vectore_store import ChromaVectorStore
from src.search import RAGSearch

# Example
if __name__ == "__main__":
    documents = load_all_document("./data")
    chunks = EmbeddingManager().chunk_document(documents)
    chunk_vectors = EmbeddingManager().generate_embeddings(chunks)
    print(chunk_vectors)
    ChromaVectorStore().add_documents(chunk_document)

    rag_search = RAGSearch();
    query="What is Attention?"
    response = rag_search.chat(query)


