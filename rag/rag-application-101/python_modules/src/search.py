from langchain_groq import ChatGroq
import os
from src.embedding import EmbeddingManager
from src.vectore_store import ChromaVectorStore


class RAGSearch:
    """ handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: ChromaVectorStore, embedding_manager: EmbeddingManager):
    """
    Initialize the RAG retriever

    Args:
        vector_store: ChromaDB vector store
        embedding_manager: Embedding manager
    """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # Get the GROQ LLM API key from Environment variable setting
        load_dotenv()  # loads .env file
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env or environment.")
        
        # Initialize the GRQ LLM with a supported model
        this.llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=groq_api_key, temperature=0.1)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float=0.0) -> List[Dict[str, Any]]:
    """
    Retrieve documents based on a query

    Args:
        query: Query string
        top_k: Number of documents to retrieve

    Returns:
        List of retrieved documents
    """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")
        try:
            # Generate embeddings for the query
            query_embedding = self.embedding_manager.generate_embeddings([query])
            #print(f"Query embedding generated: {query_embedding}")

            # search in vector store
            print(f"Searching for documents in vector store...")
            results = self.vector_store.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k

            )
            print(f"Found {len(results['documents'])} documents in the vector store")
            #print(f"Results: {results}")

            #process results
            retrieved_documents = []

            for i, document in enumerate(results['documents'][0]):

            score = results['distances'][0][i]
            if score < score_threshold:
                continue
            metadata = results['metadatas'][0][i]
            ids = results['ids'][0][i]

            retrieved_documents.append({
                'id': ids,
                'page_content': document,
                'metadata': metadata,
                'similarity_score': 1 - score,
                'distance': score,
                'rank': i + 1
            })
            print(f"Successfully retrieved {len(retrieved_documents)} documents")
            return retrieved_documents

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise
    def chat(query):
        # Simple RAG function: retrieval context + generate response

        # get the retriever/Context from query
        retriever = this.retrieve(query)

        context = "\n".join([doc['page_content'] for doc in retriver])
        if not context:
            return "No relevant context found to answer the question"

        # generate response using GROQ LLM
        prompt=f"""Use the following pieces of context to answer the question concisely.
        Context:
        {context}
        Question: {query}
        Answer:"""
        response = llm.invoke([prompt.format(context=context, question = query)])
        return response.content
