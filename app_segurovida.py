import streamlit as st
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="Asistente de Seguros")
st.title("🛡️ Agente Virtual - Tú Eliges Vida")

@st.cache_resource
def crear_agente():
    loader = TextLoader("tu_eliges_vida.txt")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

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
