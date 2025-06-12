import streamlit as st
import time
import os, sys

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pathlib import Path

# Añade el directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.multiagent_manager import create_graph

st.set_page_config(layout="wide")
st.title('Recommender System')

# Compilamos el grafo una vez al inicio
recommendation_graph = create_graph()

# Inicialización del historial de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

MAX_HISTORY = 20
CONTEXT_SIZE = 5000

# Plantilla de prompt simplificada
prompt_template = PromptTemplate(
    input_variables=["history", "human_input"],
    template="{history}\nUser: {human_input}\nAssistant:"
)

# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Trim Function (Removes Oldest Messages) ---- #
def trim_memory():
    while len(st.session_state.chat_history) > MAX_HISTORY * 2:
        st.session_state.chat_history.pop(0)
        if st.session_state.chat_history:
            st.session_state.chat_history.pop(0)

# ---- Handle User Input ---- #
if prompt := st.chat_input("Say something"):
    # Mostrar input del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Recortar historial antes de generar respuesta
    trim_memory()

    # Ejecutar el grafo de recomendación
    inputs = {"input": prompt}
    result = recommendation_graph.invoke(inputs)
    
    full_response = result["response"]

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.markdown(full_response)

    # Almacenar respuesta en el historial
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Recortar historial después de almacenar la respuesta
    trim_memory()