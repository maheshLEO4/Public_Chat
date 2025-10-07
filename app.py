import streamlit as st
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from lib
from lib.config import get_mongodb_uri
from lib.ai.query_processor import process_bot_query

# MongoDB connection
def get_mongodb_client():
    mongodb_uri = get_mongodb_uri()
    print(f"Connecting to MongoDB with URI: {mongodb_uri[:50]}...")  # Debug
    return MongoClient(mongodb_uri)

def get_bot_config(bot_id):
    """Get bot configuration from MongoDB"""
    print(f"Looking for bot_id: {bot_id}")  # Debug
    client = get_mongodb_client()
    db = client.chatbot_builder
    
    # List all collections for debugging
    collections = db.list_collection_names()
    print(f"Available collections: {collections}")  # Debug
    
    # Check if chatbots collection exists and has data
    if 'chatbots' in collections:
        bot_count = db.chatbots.count_documents({})
        print(f"Total bots in database: {bot_count}")  # Debug
        
        # Get all bot IDs for debugging
        all_bots = list(db.chatbots.find({}, {'bot_id': 1, 'name': 1}))
        print(f"All bots: {[(bot.get('bot_id'), bot.get('name')) for bot in all_bots]}")  # Debug
    
    bot_config = db.chatbots.find_one({'bot_id': bot_id})
    print(f"Found bot config: {bot_config is not None}")  # Debug
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
    query_params = st.query_params
    bot_id = query_params.get("bot_id", [None])[0]
    
    st.write(f"Debug: Received bot_id = {bot_id}")  # Debug
    
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
        
        **Debug Info:**
        - Bot ID: {bot_id}
        - Make sure this bot exists in your builder app
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
                # Process the query using your existing RAG system
                result = process_bot_query(
                    user_id=bot_config['user_id'],
                    bot_id=bot_id,
                    query=prompt,
                    system_prompt=bot_config.get('system_prompt', ''),
                    temperature=bot_config.get('temperature', 0.7)
                )
                
                if result['success']:
                    # Display the response
                    st.markdown(result['answer'])
                    
                    # Show sources if available
                    if result.get('sources'):
                        with st.expander("📚 Sources"):
                            for source in result['sources']:
                                st.write(f"**{source['document']}** - Page {source['page']}")
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result['answer']
                    })
                    
                    # Log the chat session
                    log_chat_session(bot_id, prompt, result['answer'])
                    
                else:
                    # Error handling
                    error_msg = result.get('error', 'Sorry, I encountered an error processing your question.')
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