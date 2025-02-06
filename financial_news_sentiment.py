import streamlit as st
import os
import base64
import torch
import requests
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from yahoo_fin import stock_info as si

st.set_page_config(layout="wide")
device = torch.device('cpu')

# Load FinBERT Model for Financial Sentiment Analysis
checkpoint = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
finbert = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sentiment_pipeline = pipeline("text-classification", model=finbert, tokenizer=tokenizer)

# Fetch stock price data
def get_stock_data(ticker):
    try:
        stock_price = si.get_live_price(ticker)
        return stock_price
    except Exception as e:
        return f"Error fetching stock price: {e}"

# Ingest Financial PDFs
@st.cache_resource
def data_ingestion():
    loader = PyPDFLoader("docs/tm2412112d4_ars.pdf")



    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store embeddings in ChromaDB
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")
    return db

# Sentiment Analysis on Financial Text
def analyze_sentiment(text, chunk_size=500):
    # Split text into chunks of `chunk_size`
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    results = []
    for chunk in chunks:
        result = sentiment_pipeline(chunk)
        results.append(result[0])  # Append first result of each chunk

    # Aggregate results (majority voting or average confidence)
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_confidence = 0
    
    for res in results:
        sentiment_counts[res["label"].lower()] += 1
        total_confidence += res["score"]

    # Determine final sentiment
    final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    avg_confidence = total_confidence / len(results)

    return final_sentiment, avg_confidence


# Process and display sentiment results
def process_financial_insights(pdf_text, stock_ticker):
    sentiment, confidence = analyze_sentiment(pdf_text)
    stock_price = get_stock_data(stock_ticker)

    result = f"""
    **Sentiment Analysis on Financial Document**  
    - Sentiment: **{sentiment}**  
    - Confidence: **{confidence:.2f}**  
    - Stock Price ({stock_ticker}): **${stock_price:.2f}**  
    """
    return result

# Streamlit UI
def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>📈 AI-Powered Financial News Sentiment Analyzer</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a financial report (PDF)", type=["pdf"])
    ticker = st.text_input("Enter a stock ticker (e.g., AAPL, TSLA, MSFT)")
    
    if uploaded_file and ticker:
        file_path = f"./docs/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load document
        db = data_ingestion()
        
        # Extract text from the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_content = " ".join([doc.page_content for doc in documents])
        
        # Get sentiment and stock data
        result = process_financial_insights(text_content, ticker)
        
        st.markdown(result, unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()
