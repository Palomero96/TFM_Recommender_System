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
        self.retriever = MovieRetriever(collection_name="Pruebas") 
        
        # Definimos el prompt que le pasaremos a nuestro modelo LLM para que realize la tarea que se le solicita
        self.prompt = PromptTemplate.from_template("""
            Eres un critico de cine experto en recomendar libros a los usuarios basandote en sus preferencias.
            Tienes que dar una recomendación basandote en el input que recibas del usuario (recibiras un libro) que te diga el usuario 
            y teniendo en cuenta el contexto que tienes con la información de libros.
            
            Ten en cuenta que el contexto que se te va a dar esta en ingles y el usuario puede preguntar en cualquier idioma.
            Contexto: {context}
                                            
            La pregunta del usuario puede estar en cualquier idioma. Tienes que responder en el idioma que te pregunte el usuario.
            Pregunta del usuario: {input}

            Recomienda una única película al usuario utilizando la siguiente estructura de salida:                                          
            🎬 Película recomendada: [Título exacto de la película]  
            🗓️ Año de estreno: [Año] (si disponible)  
            🌐 Idioma original: [Idioma] (si disponible)  
            🎭 Género: [Género]  
            🌟 Puntuación: [X.X/5] (si disponible)

            Justifica brevemente por qué esta película es adecuada según la descripción de la película y el siguiente input proporcionado por el usuario:  
            {input}

            La justificación debe ser concisa (máximo 2-3 frases) y centrarse en las similitudes o elementos que encajan con los gustos o intereses del usuario.

            Justificación: [2-3 frases sobre similitudes con el libro de referencia o el input]

            Reglas:
            1. Usa SOLO información de las peliculas proporcionados.
            2. Si no hay peliculas similares, di: "No encontré coincidencias precisas".
            3. Mantén la respuesta concisa y profesional.
            4. Responde en el idioma en el que ha preguntado el usuario.
                            
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
    print(agent.recommend_movie("Pelicula similar a como Cien años de soledad"))
