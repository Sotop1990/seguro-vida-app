import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# API KEY desde variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

# UI básica
st.set_page_config(page_title="Asistente de Seguros")
st.title("🛡️ Agente Virtual - Tú Eliges Vida")

@st.cache_resource
def crear_agente():
    loader = TextLoader("tu_eliges_vida.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return chain

agente = crear_agente()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

consulta = st.text_input("¿En qué puedo ayudarte sobre el seguro Tú Eliges Vida?")
if consulta:
    respuesta = agente.run(consulta)
    st.session_state.chat_history.append(("Tú", consulta))
    st.session_state.chat_history.append(("Agente", respuesta))

for rol, mensaje in reversed(st.session_state.chat_history):
    st.markdown(f"**{rol}:** {mensaje}")