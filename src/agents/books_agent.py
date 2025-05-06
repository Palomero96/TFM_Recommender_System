from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

class BooksAgent:
    def __init__(self):
        self.llm = Ollama(
            model="qwen2:5",  # Ajusta según tu modelo
            base_url=OLLAMA_BASE_URL,
            temperature=0.3  # Menor aleatoriedad para decisiones críticas
        )
        self.prompt = PromptTemplate.from_template("""
            Eres un agente director experto en clasificar consultas sobre recomendaciones.
            Determina si el usuario busca:
            - "libro" (recomendación de libros basados en un libro).
            - "película" (recomendación de películas basadas en un libro).
            - "serie" (recomendación de series basadas en un libro).
            
            Pregunta del usuario: {input}
            
            Responde SOLO con una de estas tres opciones: "libro", "película" o "serie".
            No des explicaciones ni detalles.
            """)

    def analyze_query(self, state: dict) -> dict:
        """Clasifica la consulta y devuelve la decisión."""
        chain = self.prompt | self.llm
        decision = chain.invoke({"input": state["input"]}).strip().lower()
        return {"decision": decision, "input": state["input"]}