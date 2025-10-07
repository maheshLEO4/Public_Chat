import os
import streamlit as st
from dotenv import load_dotenv

def get_api_key(api_key_name):
    """
    Get API key from environment variables or Streamlit secrets
    """
    # Load .env file for local development
    load_dotenv()
    
    # Try Streamlit secrets first (for deployment)
    try:
        if hasattr(st, 'secrets') and st.secrets:
            if api_key_name in st.secrets:
                return st.secrets.get(api_key_name)
    except Exception:
        pass
    
    # Try environment variable (for local development)
    if api_key_name in os.environ:
        return os.environ[api_key_name]
    
    # Return None if not found
    return None

def validate_api_key():
    """
    Validate that GROQ API key exists and return it
    """
    api_key = get_api_key('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return api_key

def get_qdrant_config():
    """
    Get Qdrant Cloud configuration
    """
    return {
        'api_key': get_api_key('QDRANT_API_KEY'),
        'url': get_api_key('QDRANT_URL')
    }

def get_mongodb_uri():
    """
    Get MongoDB connection URI
    """
    return get_api_key('MONGODB_URI')