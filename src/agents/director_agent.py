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
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )
        self.prompt = PromptTemplate.from_template("""
            Eres un agente director experto en clasificar consultas sobre recomendaciones de libros y peliculas. El usuario te pedirá recomendaciones en base a libros y generos
            Determina si el usuario busca:
            - 
            - Recomendaciones de libro en base a libros, peliculas o generos. Esta categoría se definirá como "libro".
            - Recomendaciones de peliculas en base a libros, peliculas o generos. Esta categoría se definirá como "pelicula".
            - Si no te encaja en ninguna de estas opciones anteriores indica que es una respuesta "general"                                       
            
            La pregunta que te haga el usuario puede estar  en cualquier idioma.
            Pregunta del usuario: {input}
            
            Responde SOLO con una de estas tres opciones: "libro", "pelicula", o "general".
            No des explicaciones ni detalles.
            """)

    def analyze_query(self, state: dict) -> dict:
        """Clasifica la consulta y devuelve la decisión."""
        #TODO: Añadir la parte de ejemplos
        # Genera el texto del prompt
        prompt_text = self.prompt.format(input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        decision = self.llm.invoke(prompt_text)
        # Devuelve el estado original con la decisión añadida
        return {**state, "decision": decision}

