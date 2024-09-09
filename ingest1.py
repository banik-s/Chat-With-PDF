from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import os

# Initialize Chroma client
chroma_client = chromadb.Client()
persist_directory = "db"

def main():
    collection_name = "my_collection"
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Load and process documents
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load()
                
                # Print the raw content of the documents
                print(f"Loaded documents: {documents}")

                print("Splitting into chunks")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)

                # Extract text content from Document objects
                text_content = [text.page_content for text in texts]
                print(f"Number of texts: {len(text_content)}")
                print(f"Texts: {text_content}")

                # Generate unique IDs for each text
                ids = [f"id_{i}" for i in range(len(text_content))]
                print(f"Generated IDs: {ids}")

                # Create embeddings
                print("Loading sentence transformers model")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

                # Add documents to the collection
                if text_content and ids:  # Ensure lists are not empty
                    print(f"Adding documents to the collection...")
                    try:
                        collection.upsert(
                            documents=text_content,
                            ids=ids
                        )
                    except Exception as e:
                        print(f"Error during upsert: {e}")
                else:
                    print("No documents or IDs to add.")

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()
