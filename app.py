import streamlit as st
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URI = os.getenv('MONGODB_URI')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# MongoDB connection
def get_mongodb_client():
    return MongoClient(MONGODB_URI)

def get_bot_config(bot_id):
    """Get bot configuration from MongoDB"""
    client = get_mongodb_client()
    db = client.chatbot_builder
    bot_config = db.chatbots.find_one({'bot_id': bot_id})
    client.close()
    return bot_config

def log_chat_session(bot_id, user_message, bot_response):
    """Log chat session to database"""
    try:
        client = get_mongodb_client()
        db = client.chatbot_builder
        db.chat_sessions.insert_one({
            'bot_id': bot_id,
            'user_message': user_message,
            'bot_response': bot_response,
            'timestamp': datetime.utcnow(),
            'source': 'public_chat'
        })
        client.close()
    except Exception as e:
        print(f"Error logging chat: {e}")

def get_vector_search_results(bot_id, query):
    """Search Qdrant for relevant documents"""
    try:
        # Search Qdrant vector database
        response = requests.post(
            f"{QDRANT_URL}/collections/chatbot_{bot_id}/points/search",
            headers={"api-key": QDRANT_API_KEY},
            json={
                "vector": [0] * 384,  # Placeholder - you'd need proper embedding
                "limit": 5,
                "with_payload": True
            }
        )
        if response.status_code == 200:
            return response.json().get('result', [])
        return []
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

def generate_ai_response(query, context, bot_config):
    """Generate AI response using Groq API"""
    try:
        system_prompt = bot_config.get('system_prompt', 'You are a helpful AI assistant.')
        temperature = bot_config.get('temperature', 0.7)
        
        # Prepare context from vector search results
        context_text = ""
        if context:
            context_text = "Relevant information:\n" + "\n".join([
                f"- {item['payload'].get('text', '')[:200]}..." 
                for item in context[:3]  # Use top 3 results
            ])
        
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n{context_text}\n\nInstructions: Use the provided context to answer the question. If the context doesn't contain relevant information, say you don't know based on your knowledge base."
            },
            {
                "role": "user", 
                "content": query
            }
        ]
        
        # Call Groq API
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": messages,
                "model": "llama-3.1-8b-instant",
                "temperature": temperature,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
            
    except Exception as e:
        print(f"AI response error: {e}")
        return "I'm experiencing technical difficulties. Please try again in a moment."

def process_user_query(bot_id, query, bot_config):
    """Process user query with RAG pipeline"""
    try:
        # Step 1: Search vector database for relevant context
        search_results = get_vector_search_results(bot_id, query)
        
        # Step 2: Generate AI response with context
        response = generate_ai_response(query, search_results, bot_config)
        
        return {
            'success': True,
            'response': response,
            'sources_used': len(search_results)
        }
        
    except Exception as e:
        print(f"Query processing error: {e}")
        return {
            'success': False,
            'response': "Sorry, I encountered an error processing your question.",
            'sources_used': 0
        }

def main():
    st.set_page_config(
        page_title="ChatBot",
        page_icon="🤖", 
        layout="centered"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 12px;
        border-radius: 15px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get bot_id from query parameters
    query_params = st.experimental_get_query_params()
    bot_id = query_params.get("bot_id", [None])[0]
    
    if not bot_id:
        st.error("""
        ## No chatbot specified! 
        
        Please use a valid chat URL provided by the bot creator.
        
        If you're looking to create your own chatbot, visit our builder app.
        """)
        st.stop()
    
    # Get bot configuration
    with st.spinner("Loading chatbot..."):
        bot_config = get_bot_config(bot_id)
    
    if not bot_config:
        st.error("""
        ## Chatbot Not Found
        
        The chatbot you're trying to access doesn't exist or has been removed.
        Please check the URL or contact the bot creator.
        """)
        st.stop()
    
    if not bot_config.get('is_active', True):
        st.error("""
        ## Chatbot Inactive
        
        This chatbot is currently inactive. 
        The creator may be updating it or has temporarily disabled it.
        """)
        st.stop()
    
    # Display bot header
    st.title(f"💬 {bot_config['name']}")
    
    # Bot description if available
    if bot_config.get('description'):
        st.write(bot_config['description'])
    
    st.caption("Ask me anything about my knowledge base!")
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = bot_config.get('welcome_message', "Hello! I'm here to help answer your questions based on my knowledge base. What would you like to know?")
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                # Process the query
                result = process_user_query(bot_id, prompt, bot_config)
                
                if result['success']:
                    # Display the response
                    st.markdown(result['response'])
                    
                    # Show sources info if available
                    if result['sources_used'] > 0:
                        st.caption(f"📚 Used {result['sources_used']} knowledge sources")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['response']
                    })
                    
                    # Log the chat session
                    log_chat_session(bot_id, prompt, result['response'])
                    
                else:
                    # Error handling
                    error_msg = "I apologize, but I'm having trouble accessing my knowledge base right now. Please try again in a moment."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Footer
    st.markdown("---")
    st.caption("Powered by AI • ChatBot Builder")

if __name__ == "__main__":
    main()