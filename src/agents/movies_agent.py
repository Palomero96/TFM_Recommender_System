from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from src.agents.retriever import  MovieRetriever

import os
from dotenv import load_dotenv
load_dotenv()  # Carga .env


class MoviesAgent:
    def __init__(self):
        # Definimos el modelo LLM que usaremos
        self.llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL"),
            base_url=os.getenv("OLLAMA_BASE_URL"),
            temperature=os.getenv("OLLAMA_TEMPERATURE")
        )
        # Definimos el retriever que usaremos indicando la coleccion en la que buscara
        self.retriever = MovieRetriever(collection_name="movies") 
        
        # Definimos el prompt que le pasaremos a nuestro modelo LLM para que realize la tarea que se le solicita
        self.prompt = PromptTemplate.from_template("""
            Eres un critico de cine experto en recomendar libros a los usuarios basandote en sus preferencias.
            Tienes que dar una recomendación basandote en el input que recibas del usuario (recibiras un libro) que te diga el usuario 
            y teniendo en cuenta el contexto que tienes con la información de libros.
            
            Ten en cuenta que el contexto que se te va a dar esta en ingles y el usuario puede preguntar en cualquier idioma.
            Contexto: {context}
                                                   
            Reglas:
            1. Usa SOLO información de las peliculas proporcionados.
            2. Si no hay peliculas similares, di: "No encontré coincidencias precisas".
            3. Mantén la respuesta concisa y profesional.
            4. Tienes que responder obligatoriamente en el idioma que te hable el usuario. Si te habla en ingles respondes en ingles.                               

            Recomienda una única película al usuario utilizando la siguiente estructura de salida:                                          
            🎬 Película recomendada: [Título exacto de la película]  
            🗓️ Año de estreno: [Year]   
            🌐 Idioma original: [Original_language]  
            🎭 Género: [Género]  
            🌟 Puntuación: [X.X/5] [Rating] 
                                                   
            Justificación: [Párrafo de justificación de la recomendación]


            Justifica la recomendación en un breve párrafo. Habla únicamente de la pelicula que vas a recomendar, basándote en su descripción, género y
            los datos que el usuario ha proporcionado en su consulta. La consulta del usuario es la siguiente:  
            {input}

            Ten en cuenta si la pelicula tiene una puntuación alta, si pertenece a un género que encaje con los gustos del usuario, o si su descripción lo hace especialmente relevante para su interés.

            """)
        
      
    def recommend_movie(self,  state: dict) -> dict:
        # Obtiene el contexto formateado desde Milvus usando el retriever
        context = self.retriever(state['input'])
        
        # Genera el texto del prompt
        prompt_text = self.prompt.format(context=context, input=state['input'])
        # Pasa el prompt al LLM y devuelve la respuesta
        response = self.llm.invoke(prompt_text)
        # Devolvemos la respuesta
        return {**state, "output": response}



if __name__ == "__main__":
    agent = MoviesAgent()
    initial_state = {
            "input": "Estoy buscando una pelicula que tenga que ver con la historia",
            "output": "",
            "context": "",
            "decision": ""
        }
    print(agent.recommend_movie(initial_state))
