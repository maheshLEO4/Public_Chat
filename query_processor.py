import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from vector_store import get_vector_store
from config import validate_api_key

@st.cache_resource(show_spinner=False)
def get_cached_qa_chain(groq_api_key, user_id, bot_id, system_prompt, temperature):
    """Cached QA chain - only loads once per user session"""
    try:
        print(f"🔍 Creating QA chain for user: {user_id}, bot: {bot_id}")
        db = get_vector_store(user_id, bot_id)
        
        if db is None:
            print("❌ Vector store is None - knowledge base not found")
            return None

        print("✅ Vector store loaded successfully")
        
        # Simple, effective prompt
        CUSTOM_PROMPT_TEMPLATE = f"""
        {system_prompt}

        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {{context}}
        Question: {{question}}

        Start the answer directly. No small talk please.
        """
        
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )

        # Simple retriever
        retriever = db.as_retriever(
            search_kwargs={"k": 5}  # Get top 5 relevant documents
        )
        
        # LLM config
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=temperature,
            groq_api_key=groq_api_key,
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        
        print("✅ QA chain created successfully")
        return qa_chain
        
    except Exception as e:
        print(f"❌ Error creating QA chain: {e}")
        return None

def format_source_documents(source_documents):
    """Format source documents for display."""
    formatted_sources = []
    
    for doc in source_documents:
        try:
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            
            # Determine source type and name
            if isinstance(source, str) and source.startswith(('http://', 'https://')):
                source_type = 'web'
                source_name = source
            else:
                source_type = 'pdf'
                source_name = os.path.basename(str(source)) if source else 'Unknown'
            
            # Get page number
            page_num = metadata.get('page', 'N/A')
            if isinstance(page_num, int):
                page_num += 1  # Make it 1-indexed for display
            
            # Create excerpt
            excerpt = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            formatted_sources.append({
                'document': source_name,
                'page': page_num,
                'excerpt': excerpt,
                'type': source_type
            })
            
        except Exception as e:
            print(f"Error formatting source: {e}")
            continue
    
    return formatted_sources

def process_bot_query(user_id, bot_id, query, system_prompt, temperature):
    """Process user query and return answer with sources."""
    try:
        print(f"🔍 Processing query for user: {user_id}, bot: {bot_id}")
        print(f"🔍 Query: {query}")
        
        groq_api_key = validate_api_key()
        print("✅ Groq API key validated")
        
        # Get cached chain
        qa_chain = get_cached_qa_chain(groq_api_key, user_id, bot_id, system_prompt, temperature)
        
        if not qa_chain:
            print("❌ QA chain is None - returning knowledge base error")
            return {
                'success': False,
                'error': "Knowledge base not ready. Please add documents first."
            }
        
        print("✅ QA chain loaded - processing query...")
        
        # Process query
        response = qa_chain.invoke({'query': query})
        answer = response.get("result", "No answer generated.")
        source_documents = response.get("source_documents", [])
        
        print(f"✅ Query processed successfully")
        print(f"✅ Answer length: {len(answer)} characters")
        print(f"✅ Sources found: {len(source_documents)}")
        
        # Format sources
        formatted_sources = format_source_documents(source_documents)
        
        return {
            'success': True,
            'answer': answer,
            'sources': formatted_sources
        }
            
    except Exception as e:
        # Error handling
        error_msg = "Sorry, I encountered an issue processing your question. Please try again."
        
        if "timeout" in str(e).lower():
            error_msg = "Request timed out. Please try a shorter question."
        elif "rate limit" in str(e).lower():
            error_msg = "Rate limit exceeded. Please wait a moment and try again."
        elif "api key" in str(e).lower():
            error_msg = "API configuration issue. Please check your settings."
            
        print(f"❌ Query processing error: {e}")
        return {
            'success': False,
            'error': error_msg
        }