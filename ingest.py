from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from chromadb import Client

def main():
    # Load documents
    pdf_loader = PyPDFLoader(r'C:\Users\warden\Desktop\Chat-With-PDF\docs\fastfacts-what-is-climate-change.pdf')
    documents = pdf_loader.load()
    
    # Create a Chroma client (new setup)
    chroma_client = Client()

    # Create or retrieve a collection
    collection = chroma_client.get_or_create_collection(name="pdf_collection")

    # Load embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Upsert documents with embeddings
    collection.upsert(
        documents=[doc.page_content for doc in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    # Query the collection (example query)
    results = collection.query(
        query_texts=["This is a document about climate change"], 
        n_results=2
    )
    
    print(results)

if __name__ == "__main__":
    main()
