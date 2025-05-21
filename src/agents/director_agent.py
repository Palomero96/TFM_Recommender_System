from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class DirectorAgent:
    def __init__(self):
        self.llm = Ollama(
            model=OLLAMA_MODEL,  # Ajusta según tu modelo
            base_url=OLLAMA_BASE_URL,
            temperature=0.3  # Menor aleatoriedad para decisiones críticas
        )
        self.prompt = PromptTemplate.from_template("""
            Eres un agente director experto en clasificar consultas sobre recomendaciones de libros y peliculas en función de libros o generos.
            Determina si el usuario busca:
            - Recomendaciones de libro en base a un libro. Esta categoría se definirá como "libro".
            - Recomendaciones de libro en base a generos. Esta categoría se definirá como "librogenero".
            - Recomendaciones de peliculas en base a un libro. Esta categoría se definirá como "pelicula".
            - Recomendaciones de peliculas en base a generos. Esta categoría se definirá como "peliculagenero".
            - Si no te encaja en ninguna de estas opciones anteriores indica que es una respuesta "general"                                       
            
            Pregunta del usuario: {input}
            
            Responde SOLO con una de estas tres opciones: "libro", "librogenero", "pelicula", "peliculagenero" o "general".
            No des explicaciones ni detalles.
            """)

    def analyze_query(self, state: dict) -> dict:
        """Clasifica la consulta y devuelve la decisión."""
        chain = self.prompt | self.llm
        decision = chain.invoke({"input": state["input"]}).strip().lower()
        return {"decision": decision, "input": state["input"]}

