from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pandas as pd
import os

# Cargamos las variables de entorno
from dotenv import load_dotenv
load_dotenv()  # Carga .env

from pathlib import Path
# Cargamos la ruta de los ejemplos que le vamos a pasar al agente director para que le sea mas facil decidir
ROOT_DIR = Path(__file__).resolve().parents[2]
EJEMPLOS_PATH = ROOT_DIR / "data" / "ejemplos.txt"

with open(EJEMPLOS_PATH, "r", encoding="utf-8") as f:
    EJEMPLOS = f.read()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


class DirectorAgent:
    def __init__(self):
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )

        self.prompt = PromptTemplate.from_template("""
            Eres un agente director experto en clasificar consultas sobre recomendaciones de libros y peliculas. El usuario te pedirá recomendaciones en base a libros y generos
            Determina si el usuario busca:
            - Recomendaciones de libro en base a libros, peliculas o generos. Esta categoría se definirá como "libro".
            - Recomendaciones de peliculas en base a libros, peliculas o generos. Esta categoría se definirá como "pelicula".
            - Si no te encaja en ninguna de estas opciones anteriores indica que es una respuesta "general"                                       
            
            Te voy a facilitar una serie de ejemplos con clasificaciones para que te sea mas facil determinar que busca el usuario.
            Los ejemplos constan de dos columnas, una es input que simula la pregunta del usuario y otra label con la clasificacion de la pregunta.
            Ejemplos: {ejemplos}
                                                                                        
            La pregunta que te haga el usuario puede estar  en cualquier idioma.
            Pregunta del usuario: {input}
            
            Responde SOLO con una de estas tres opciones: "libro", "pelicula", o "general".
            No des explicaciones ni detalles.
            """)

    def analyze_query(self, state: dict) -> dict:
        """Clasifica la consulta y devuelve la decisión."""
        # Genera el texto del prompt
        prompt_text = self.prompt.format(ejemplos=EJEMPLOS,input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        decision = self.llm.invoke(prompt_text)
        # Devuelve el estado original con la decisión añadida
        return {**state, "decision": decision}

if __name__ == "__main__":
    agent = DirectorAgent()
    print(agent.analyze_query({"input": "Libro similar a como Cien años de soledad"}))
    print(agent.analyze_query({"input": "¿Quién escribió Cien años de soledad"}))

    