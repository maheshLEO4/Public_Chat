import streamlit as st
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from lib
from config import get_mongodb_uri
from query_processor import process_bot_query

# MongoDB connection with proper error handling
def get_mongodb_client():
    try:
        mongodb_uri = get_mongodb_uri()
        if not mongodb_uri:
            st.error("MongoDB URI not found. Please check your environment variables.")
            return None
        
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        st.error(f"MongoDB connection failed: {str(e)}")
        return None

def get_bot_config(bot_id):
    """Get bot configuration from MongoDB"""
    try:
        client = get_mongodb_client()
        if not client:
            return None
            
        db = client.chatbot_builder
        print(f"🔍 Searching for bot_id: {bot_id}")
        
        # Get the specific bot
        bot_config = db.chatbots.find_one({'bot_id': bot_id})
        print(f"✅ Bot found: {bot_config is not None}")
        
        client.close()
        return bot_config
        
    except Exception as e:
        print(f"❌ Error getting bot config: {e}")
        return None

def log_chat_session(bot_id, user_message, bot_response):
    """Log chat session to database"""
    try:
        client = get_mongodb_client()
        if client:
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
    
    # ✅ FIXED: Proper query parameter extraction
    query_params = st.query_params
    bot_id = query_params.get("bot_id", "")
    
    # If it's a list, take the first element, otherwise use the string directly
    if isinstance(bot_id, list):
        bot_id = bot_id[0] if bot_id else ""
    
    st.info(f"🔄 Loading chatbot with ID: `{bot_id}`")
    
    if not bot_id:
        st.error("""
        ## No chatbot specified! 
        
        Please use a valid chat URL with ?bot_id= parameter.
        Example: https://public-chat-app.streamlit.app/?bot_id=572eb353
        """)
        st.stop()
    
    # Test MongoDB connection first
    with st.spinner("Connecting to database..."):
        client = get_mongodb_client()
        if not client:
            st.error("""
            ## Database Connection Failed
            
            Cannot connect to MongoDB. Please check:
            - MongoDB URI in environment variables
            - Network connectivity
            - Database permissions
            """)
            st.stop()
        else:
            client.close()
    
    # Get bot configuration
    with st.spinner("Loading chatbot configuration..."):
        bot_config = get_bot_config(bot_id)
    
    if not bot_config:
        st.error(f"""
        ## Chatbot Not Found
        
        The chatbot with ID `{bot_id}` was not found in the database.
        
        **Please check:**
        - The bot ID is correct: `{bot_id}`
        - The bot exists in your builder app
        - The bot is active
        
        **Try this exact URL:** https://public-chat-app.streamlit.app/?bot_id=572eb353
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
    
    st.success("✅ Chatbot loaded successfully!")
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