import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from config import get_qdrant_config

def get_bot_collection_name(user_id, bot_id):
    """Get bot-specific Qdrant collection name"""
    collection_name = f"chatbot_{user_id}_{bot_id}"
    print(f"🔍 Qdrant collection name: {collection_name}")
    return collection_name

@st.cache_resource
def get_qdrant_client():
    """Cached Qdrant client"""
    qdrant_config = get_qdrant_config()
    print(f"🔍 Qdrant config: URL={qdrant_config['url']}, API Key length={len(qdrant_config['api_key'])}")
    return QdrantClient(
        url=qdrant_config['url'],
        api_key=qdrant_config['api_key'],
        timeout=30
    )

def check_sentence_transformers():
    """Check if sentence-transformers is available"""
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers is properly installed")
        return True
    except ImportError as e:
        print(f"❌ sentence-transformers import error: {e}")
        return False

def get_embedding_model():
    """Get embedding model with fallback options"""
    try:
        # Try new import first
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ Using langchain_huggingface embeddings")
    except ImportError:
        try:
            # Fallback to community import
            from langchain_community.embeddings import HuggingFaceEmbeddings
            print("✅ Using langchain_community embeddings")
        except ImportError as e:
            print(f"❌ Could not import HuggingFaceEmbeddings: {e}")
            return None
    
    try:
        # Verify sentence-transformers is working
        if not check_sentence_transformers():
            return None
            
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        print(f"❌ Error creating embedding model: {e}")
        return None

def get_vector_store(user_id, bot_id):
    """Get Qdrant vector store for specific bot"""
    collection_name = get_bot_collection_name(user_id, bot_id)
    
    qdrant_config = get_qdrant_config()
    if not qdrant_config['api_key'] or not qdrant_config['url']:
        raise ValueError("Qdrant Cloud not configured")
    
    try:
        client = get_qdrant_client()
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            print(f"✅ Qdrant collection exists: {collection_name}")
            print(f"📊 Collection points: {collection_info.points_count}")
        except Exception as e:
            print(f"❌ Qdrant collection not found: {collection_name}")
            print(f"❌ Error: {e}")
            return None
        
        # Get embedding model
        embedding_model = get_embedding_model()
        if embedding_model is None:
            print("❌ Failed to initialize embedding model")
            return None
        
        # Import Qdrant vector store
        try:
            from langchain_qdrant import Qdrant
            print("✅ Using langchain_qdrant")
        except ImportError:
            from langchain_community.vectorstores import Qdrant
            print("✅ Using langchain_community Qdrant")
        
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embedding_model
        )
        
        print(f"✅ Vector store created successfully for {collection_name}")
        return vector_store
        
    except Exception as e:
        print(f"❌ Error initializing Qdrant: {e}")
        return None

def create_fallback_response():
    """Create a fallback response when vector store fails"""
    return "I'm here to help! However, my knowledge base is currently unavailable. Please try again later or contact support."