import requests
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity

# WSL2 IP Address (adjust if needed)
WSL2_OLLAMA_IP = "http://localhost:11434"

# 1. Load documents
loader = TextLoader('E:/PROJECTS/docs/example.txt')  # Load single text file
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings with nomic-embed-text via Ollama
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url=WSL2_OLLAMA_IP
)

# 4. Store in Chroma vector database
db = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
retriever = db.as_retriever()

# 5. Connect to Ollama LLM in WSL2 (e.g., mistral or llama3)
llm = OllamaLLM(
    model="mistral:latest",
    base_url=WSL2_OLLAMA_IP
)

# 6. Create QA chain using RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 7. Helper function to handle missing vectors and similarity threshold
def handle_missing_vector(query):
    try:
        # Step 1: Use retriever to get relevant documents with scores
        # For Chroma with similarity scores, we need to configure the retriever
        retriever_with_scores = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.1, "k": 4}  # Lower threshold to get results
        )
        
        # Get documents with similarity scores using invoke (new method)
        results = retriever_with_scores.invoke(query)
        
        if len(results) == 0:
            return "Sorry, I couldn't find any relevant documents for your question."
        
        # Step 2: For basic similarity check, we can use the default retriever
        # and manually check if we have good results
        basic_results = retriever.invoke(query)
        
        if len(basic_results) == 0:
            return "Sorry, I couldn't find an answer for your question."
        
        # Step 3: Use QA chain to generate answer
        result = qa_chain.invoke({"query": query})
        return result["result"]
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        # Fallback: try direct QA without similarity checking
        try:
            result = qa_chain.invoke({"query": query})
            return result["result"]
        except Exception as e2:
            return f"Sorry, an error occurred: {e2}"

# Alternative approach using similarity search with scores directly
def handle_missing_vector_alternative(query):
    try:
        # Step 1: Get documents with similarity scores using similarity_search_with_score
        results_with_scores = db.similarity_search_with_score(query, k=4)
        
        if len(results_with_scores) == 0:
            return "Sorry, I couldn't find any relevant documents for your question."
        
        # Step 2: Check the similarity of the top result
        top_doc, top_score = results_with_scores[0]
        
        # Note: Different embedding models use different similarity metrics
        # For cosine similarity, scores closer to 0 are more similar
        # Adjust threshold based on your embedding model
        similarity_threshold = 1.5  # Adjust this value based on your needs
        
        if top_score > similarity_threshold:
            return f"The similarity score is too low ({top_score:.2f}). Could not find a relevant answer."
        
        # Step 3: If similarity is good enough, use QA chain
        result = qa_chain.invoke({"query": query})
        return result["result"]
        
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return f"Sorry, an error occurred: {e}"

# 8. Main loop to ask questions
while True:
    query = input("\nAsk a question (or type 'exit'): ").strip()
    if query.lower() in ("exit", "quit"):
        break
    
    # Handle empty input
    if not query:
        print("Please enter a valid question.")
        continue
    
    # Handle the query - you can choose between the two approaches
    print("Using basic approach...")
    response = handle_missing_vector(query)
    print("\nAnswer:", response)
    
    # Uncomment to try the alternative approach with explicit similarity scores
    # print("\nUsing alternative approach with similarity scores...")
    # response_alt = handle_missing_vector_alternative(query)
    # print("\nAlternative Answer:", response_alt)