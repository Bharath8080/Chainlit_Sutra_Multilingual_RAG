import os
import tempfile
import chainlit as cl
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from chainlit.input_widget import Select, Switch, Slider

# Load environment variables
load_dotenv()
api_key = os.getenv("SUTRA_API_KEY")
embedding_api_key = os.getenv("OPENAI_API_KEY")

# Define supported languages
LANGUAGES = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Punjabi", "Marathi", "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", "Japanese",
    "Arabic", "French", "German", "Spanish", "Portuguese", "Russian", "Chinese",
    "Vietnamese", "Thai", "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch",
    "Italian", "Greek", "Hebrew", "Persian", "Swedish", "Norwegian", "Danish",
    "Finnish", "Czech", "Hungarian", "Romanian", "Bulgarian", "Croatian", "Serbian",
    "Slovak", "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay", "Tagalog", "Swahili"
]

# Get regular chat model for RAG
def get_chat_model(temperature=0.7, streaming=True):
    return ChatOpenAI(
        api_key=os.getenv("SUTRA_API_KEY"),
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=temperature,
        streaming=streaming
    )

# Function to process documents
def process_documents(files, chunk_size=1000, chunk_overlap=100):
    documents = []
    pdf_elements = []
    
    for file in files:
        # Process based on file type
        if file.name.endswith(".pdf"):
            # Add to PDF elements for display
            pdf_elements.append(
                cl.Pdf(name=file.name, display="side", path=file.path)
            )
            
            # Process for RAG
            loader = PyPDFLoader(file.path)
            documents.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file.path)
            documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=embedding_api_key)
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    
    # Create conversation chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_chat_model(streaming=False),  # Use non-streaming for RAG
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain, len(document_chunks), pdf_elements

@cl.on_chat_start
async def start():
    # Initialize session data
    cl.user_session.set("documents_processed", False)
    cl.user_session.set("conversation_chain", None)
    cl.user_session.set("pdf_elements", [])
    
    # Set up chat settings
    await cl.ChatSettings(
        [
            Select(id="language", label="🌐 Language", values=LANGUAGES, initial_index=0),
            Switch(id="streaming", label="💬 Stream Response", initial=True),
            Slider(id="temperature", label="🔥 Temperature", initial=0.7, min=0, max=1, step=0.1)
        ]
    ).send()
    
    # Welcome message
    welcome_msg = """
    # 📚 Sutra Multilingual RAG 
    """
    await cl.Message(content=welcome_msg).send()
    
    # Ask for document upload
    files = await cl.AskFileMessage(
        content="Please upload PDF or DOCX files to begin!",
        accept={"application/pdf": [".pdf"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"]},
        max_size_mb=50,
        max_files=10,
        timeout=180
    ).send()
    
    if files:
        # Show processing message
        processing_msg = cl.Message(content="⏳ Processing documents...")
        await processing_msg.send()
        
        try:
            # Process documents and create conversation chain
            conversation_chain, num_chunks, pdf_elements = process_documents(files)
            cl.user_session.set("conversation_chain", conversation_chain)
            cl.user_session.set("documents_processed", True)
            cl.user_session.set("pdf_elements", pdf_elements)
            
            # Create file list for the message
            file_list = "\n".join([f"- {file.name}" for file in files])
            
            # Success message with PDF viewers for any PDF files
            success_msg = f"""
            ✅ Successfully processed {len(files)} documents into {num_chunks} chunks.
            
            """
            
            # If we have PDF elements, show them with the message
            if pdf_elements:
                await cl.Message(content=success_msg, elements=pdf_elements).send()
            else:
                await processing_msg.update(content=success_msg)
                
        except Exception as e:
            # Handle errors
            await processing_msg.update(content=f"❌ Error processing documents: {str(e)}")
            if "API key" in str(e):
                await cl.Message(content="Please check your API keys in the environment variables.").send()
    else:
        await cl.Message(content="No files uploaded. Please try again when you're ready to upload documents.").send()

@cl.on_settings_update
async def handle_settings_update(settings):
    # Just update the settings, document processing is handled at chat start
    pass

@cl.on_message
async def handle_message(msg: cl.Message):
    settings = cl.user_session.get("chat_settings", {})
    language = settings.get("language", "English")
    streaming = settings.get("streaming", True)
    temperature = settings.get("temperature", 0.7)
    
    # Check if documents have been processed
    if not cl.user_session.get("documents_processed"):
        # Ask for document upload
        await cl.Message(content="⚠️ No documents have been processed yet. Please upload documents first.").send()
        files = await cl.AskFileMessage(
            content="Please upload PDF or DOCX files to begin!",
            accept={"application/pdf": [".pdf"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"]},
            max_size_mb=20,
            max_files=10,
            timeout=180
        ).send()
        
        if files:
            # Show processing message
            processing_msg = cl.Message(content="⏳ Processing documents...")
            await processing_msg.send()
            
            try:
                # Process documents and create conversation chain
                conversation_chain, num_chunks, pdf_elements = process_documents(files)
                cl.user_session.set("conversation_chain", conversation_chain)
                cl.user_session.set("documents_processed", True)
                cl.user_session.set("pdf_elements", pdf_elements)
                
                # Create file list for the message
                file_list = "\n".join([f"- {file.name}" for file in files])
                
                # Success message
                success_msg = f"""
                ✅ Successfully processed {len(files)} documents into {num_chunks} chunks.
                
                **Processed files:**
                {file_list}
                
                Now I'll answer your question.
                """
                
                # If we have PDF elements, show them with the message
                if pdf_elements:
                    await cl.Message(content=success_msg, elements=pdf_elements).send()
                else:
                    await processing_msg.update(content=success_msg)
                    
            except Exception as e:
                await processing_msg.update(content=f"❌ Error processing documents: {str(e)}")
                return
        else:
            return
    
    conversation_chain = cl.user_session.get("conversation_chain")
    pdf_elements = cl.user_session.get("pdf_elements", [])
    
    # Create response message with PDF elements if available
    response = cl.Message(content="")
    await response.send()
    
    try:
        # Get RAG context first
        rag_response = conversation_chain.invoke(msg.content)
        context = rag_response["answer"]
        
        # Now generate a response with Sutra in the selected language
        system_prompt = f"""
        You are a helpful assistant that answers questions about documents. 
        Use the following context to answer the question.
        
        CONTEXT:
        {context}
        
        Please respond strictly in {language}.
        """
        
        chat_model = get_chat_model(temperature=temperature, streaming=streaming)
        
        # Generate response
        if streaming:
            async for chunk in chat_model.astream([
                SystemMessage(content=system_prompt),
                HumanMessage(content=msg.content)
            ]):
                await response.stream_token(chunk.content)
        else:
            result = chat_model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=msg.content)
            ])
            await response.update(content=result.content)
        
        # Add PDF elements to the response if this is the first user message
        if len(cl.user_session.get("chat_history", [])) <= 1 and pdf_elements:
            await response.update(elements=pdf_elements)
            
    except Exception as e:
        await response.update(content=f"❌ Error: {str(e)}")
        if "API key" in str(e):
            await cl.Message(content="Please check your API keys in the environment variables.").send()
