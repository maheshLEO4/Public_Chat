import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from lib.config import get_qdrant_config

def get_bot_collection_name(user_id, bot_id):
    """Get bot-specific Qdrant collection name"""
    return f"chatbot_{user_id}_{bot_id}"

@st.cache_resource
def get_qdrant_client():
    """Cached Qdrant client"""
    qdrant_config = get_qdrant_config()
    return QdrantClient(
        url=qdrant_config['url'],
        api_key=qdrant_config['api_key'],
        timeout=30
    )

def get_vector_store(user_id, bot_id):
    """Get Qdrant vector store for specific bot"""
    collection_name = get_bot_collection_name(user_id, bot_id)
    
    qdrant_config = get_qdrant_config()
    if not qdrant_config['api_key'] or not qdrant_config['url']:
        raise ValueError("Qdrant Cloud not configured")
    
    try:
        client = get_qdrant_client()
        
        # Create collection if it doesn't exist
        try:
            client.get_collection(collection_name=collection_name)
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        
        # Import here to avoid dependency issues
        try:
            from langchain_qdrant import Qdrant
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.vectorstores import Qdrant
            from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model
        )
        
        return vector_store
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        raise